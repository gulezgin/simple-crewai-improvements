import os
import sys
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
from crewai import Agent, Task, Crew, Process
from openai import AzureOpenAI
from config import OpenAIConfig, openai_config

class QueryMatchError(Exception):
    """Query matching error"""
    pass

class OpenAIConnectionError(Exception):
    """OpenAI connection error"""
    pass

client = AzureOpenAI(
    azure_endpoint=openai_config.endpoint,
    api_key=openai_config.subscription_key,
    api_version=openai_config.api_version
)

data_source_expert = Agent(
    role='Data Source Selection Expert',
    goal='Determine the most appropriate data source based on query content',
    backstory="""I am an expert in different data sources and responsible for 
    selecting the most accurate data source based on query content.""",
    allow_delegation=False,
    llm_config={
        "model": openai_config.deployment,
        "api_type": "azure",
        "temperature": 0.3,
        "client": client
    },    
    verbose=True
)

rule_analyst = Agent(
    role='Rule Analyst',
    goal='Analyze and apply data source selection rules',
    backstory="""I analyze data source selection rules and 
    determine the most appropriate rule for the query.""",
    allow_delegation=False,
    llm_config={
        "model": openai_config.deployment,
        "api_type": "azure",
        "temperature": 0.3,
        "client": client
    },    
    verbose=True
)

rules = {
    'price_rules': {
        'keywords': ['price', 'cost', 'value', 'purchase', 'sale'],
        'default_table': 'Fast Markets',
        'description': 'Used for queries related to price information, cost analysis, purchase-sale values and market pricing.',
        'example_queries': [
            "Fast markets corn price list for this month ?",
            "What are the average, low, and high prices of used cooking oil at international in different regions over time?",
            "What are the average, low, and high prices of Used Cooking Oil in the Atlantic Seaboard region over time?",
            "What is the average value of barley assessments in USD over the past 90 days, grouped by FOB details and day-time?"
        ]
    },
    'agriculture_rules': {
        'keywords': ['agriculture', 'crop', 'farm', 'yield'],
        'default_table': 'NASS Statistics',
        'description': 'Used for queries related to agricultural production, crop data, farm statistics and yield estimates published by the United States Department of Agriculture (USDA).',
        'example_queries': [
            "How many pounds of crude soybean oil stocks are stored onsite and offsite nationally by month and year?",
            "What is the weekly percentage of sorghum in excellent and good condition at the national level?",
            "What are the annual national statistics for acres of spring durum wheat harvested and planted?",
            "What is the weekly percentage of corn planted in each state?",
            "What is the weekly percentage distribution of winter wheat conditions classified as \"EXCELLENT\" or \"GOOD\" at the state level over the years?",
            "What is the annual national production of winter wheat in bushels?"
        ]
    },
    'export_rules': {
        'keywords': ['export', 'sales'],
        'default_table': 'Export Sales Report',
        'description': 'Used for queries related to export sales, foreign trade reports and international sales data.',
        'example_queries': [
            "What is the weekly progress of Barley exports in comparison to 99% of the USDA forecast and the USDA forecast minus cumulative sales for each week?",
            "What is the weekly and cumulative percentage of total U.S. corn exports over different weeks and years?",
            "What are the weekly net sales and cumulative sales for 'Wheat - SRW' over time?",
            "What are the weekly net sales and cumulative sales for all wheat, adjusted by a factor, over a given time period?",
            "What is the weekly and cumulative percentage of total barley exports for each week?"
        ]
    },
    'europe_rules': {
        'keywords': ['europe', 'eu', 'european'],
        'default_table': 'European Agricultural Statistics',
        'description': 'Used for queries containing price, production and trade data related to European Union and European countries.',
        'example_queries': [
            "How does the total wheat production in European countries compare to the total wheat production in the United States, measured in tons, over the same years?",
            "What was the production of wheat in France for the years 2016 and 2017?" 
        ]     
    },
    'trade_rules': {
        'keywords': ['trade', 'export', 'import'],
        'default_table': 'Trade Data Monitor',
        'description': 'Used for queries related to international trade flows, import-export statistics and trade balance data.',
        'example_queries': [
            "What is the seasonal export quantity and cumulative export quantity of corn (in tons) for specific countries (Argentina, Brazil, Ukraine, and the United States) over different market and calendar years and months?"
        ]
    },
    'psd_rules': {
        'keywords': ['production', 'supply', 'distribution', 'psd'],
        'default_table': 'Production, Supply, and Distribution (PSD) Statistics',
        'description': 'Used for queries related to production, supply, distribution statistics and PSD reports.',
        'example_queries': [
            "What is the weekly progress of Barley exports in comparison to 99% of the USDA forecast and the USDA forecast minus cumulative sales for each week?",
            "What is the weekly and cumulative percentage of total U.S. corn exports over different weeks and years?",
            "What is the weekly and cumulative percentage of total U.S. sorghum exports, by week, for each year?"
        ]
    }
}

# Caching class
class QueryCache:
    def __init__(self, expiry_minutes: int = 60):
        self.cache = {}
        self.expiry = expiry_minutes

    def get(self, key: str) -> Dict[str, Any]:
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(minutes=self.expiry):
                return result
            del self.cache[key]
        return None

    def set(self, key: str, value: Dict[str, Any]):
        self.cache[key] = (value, datetime.now())

query_cache = QueryCache()

# Logging settings
def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def match_query_to_rule(query: str, debug: bool = False) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    
    if not query.strip():
        raise QueryMatchError("Query cannot be empty")

    if debug:
        logger.debug(f"Query received: {query}")

    prompt = """Match the query to the most appropriate rule based on the rule descriptions below:

Rules and Descriptions:
"""
    for rule_name, rule_details in rules.items():
        prompt += f"\nRule: {rule_name}\n"
        prompt += f"Description: {rule_details['description']}\n"
        prompt += f"Keywords: {', '.join(rule_details['keywords'])}\n"
        prompt += f"Example Queries:\n"
        for example in rule_details['example_queries']:
            prompt += f"- {example}\n"
            
    prompt += f"\nQuery: {query}\n"
    prompt += "Please match the query to the most appropriate rule and explain why. Make sure to include the rule name in your response."

    try:
        completion = client.chat.completions.create(
            model=openai_config.deployment,
            messages=[
                {"role": "system", "content": "You are an AI assistant that matches user queries to appropriate rules based on semantic analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3,
        )
        
        response = completion.to_dict()
        result = response["choices"][0]["message"]["content"]

        for rule_name in rules.keys():
            if rule_name.lower() in result.lower():
                logger.info(f"Rule match successful: {rule_name}")
                return rules[rule_name]
                
    except Exception as e:
        logger.error(f"OpenAI error: {str(e)}")
        raise OpenAIConnectionError(f"OpenAI connection error: {str(e)}")
    
    return {"default_table": "Unknown Source"}

def process_with_azure_openai(query: str, debug: bool = False) -> str:
    return match_query_to_rule(query, debug)

def get_appropriate_data_source(query: str, debug: bool = False) -> str:
    logger = logging.getLogger(__name__)
    
    # Check cache
    cached_result = query_cache.get(query)
    if cached_result:
        logger.info("Result retrieved from cache")
        return cached_result["default_table"]
    
    # Rule matching
    rule_details = match_query_to_rule(query, debug)
    
    # Save result to cache
    query_cache.set(query, rule_details)
    
    if debug:
        logger.debug(f"Rule details: {rule_details}")
    
    return rule_details.get("default_table", "Data source not found")

def main():
    setup_logging(debug=True)
    logger = logging.getLogger(__name__)
    
    try:
        query = "mısırın türkiye fiyatı nedir"
        result = get_appropriate_data_source(query, debug=True)
        logger.info(f"Most appropriate data source: {result}")
        print(f"Most appropriate data source: {result}")
        
    except (QueryMatchError, OpenAIConnectionError) as e:
        logger.error(f"Process error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 