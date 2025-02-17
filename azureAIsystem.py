import openai
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class OpenAIConfig:
    endpoint: str
    deployment: str
    subscription_key: str
    api_version: str = "2024-05-01-preview"
    location: str = "francecentral"

@dataclass
class RulesConfig:
    rules: Dict[str, Any]

openai_config = OpenAIConfig(
    endpoint="https://arconarswedendev.openai.azure.com/",
    deployment="gpt-4o-1",
    subscription_key="e7876b13cf654bbeabd5892d50b6d5d5",
    api_version="2024-02-15-preview",
    location="swedencentral"
)

openai.api_type = "azure"
openai.api_base = openai_config.endpoint
openai.api_key = openai_config.subscription_key
openai.api_version = openai_config.api_version

class AgricultureSourceSelector:
    def __init__(self):
        self.sources = [
            {"name": "Eurostat International Trade in Goods", "url": "https://ec.europa.eu/eurostat/web/international-trade-in-goods", "description": "Provides statistical data on international trade in goods for the EU."},
            {"name": "Fastmarkets", "url": "https://www.fastmarkets.com/", "description": "Offers market intelligence on global commodity prices and trends."},
            {"name": "Trade Data Monitor", "url": "https://tradedatamonitor.com/", "description": "Aggregates trade data from multiple countries to monitor global trade flows."},
            {"name": "USDA ESRQuery", "url": "https://apps.fas.usda.gov/esrquery/ESRHome.aspx", "description": "Delivers export sales reporting data from the USDA."},
            {"name": "USDA Foreign Agricultural Service", "url": "https://www.fas.usda.gov/", "description": "Focuses on international trade policy and export support for U.S. agriculture."},
            {"name": "USDA National Agricultural Statistics Service", "url": "https://www.nass.usda.gov/", "description": "Provides comprehensive agricultural statistics for the U.S."}
        ]

    def recommend_source(self, query: str) -> dict:
        prompt = (
            "You are an AI system that recommends data sources based on user queries. Here is the list of sources: \n"
            + "\n".join([f"{s['name']}: {s['description']}" for s in self.sources]) + "\n\n"
            + f"User query: {query}\n"
            + "Which data source is the most relevant? Respond with the name and URL only."
        )

        response = openai.ChatCompletion.create(
            engine=openai_config.deployment,
            messages=[{"role": "system", "content": prompt}],
            temperature=0
        )

        result = response.choices[0].message['content']
        for source in self.sources:
            if source['name'] in result:
                return source
        return {"name": "Unknown", "url": "No matching source found"}

if __name__ == "__main__":
    selector = AgricultureSourceSelector()
    print("Tarım Verisi Kaynak Seçim Sistemi")
    while True:
        query = input("Sorgunuzu girin (çıkmak için 'exit' yazın): ")
        if query.lower() == "exit":
            break
        result = selector.recommend_source(query)
        print(f"Önerilen Kaynak: {result['name']} - {result['url']}")
