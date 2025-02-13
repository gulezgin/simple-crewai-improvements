from typing import Any, List, Dict
from crewai.tools import BaseTool
from ..config.config import openai_config
import openai
from pydantic import Field

class SourceSelectorTool(BaseTool):
    name: str = "Tarım Veri Kaynağı Seçici"
    description: str = "Kullanıcının sorgusu için en uygun tarım veri kaynağını seçer"
    sources: List[Dict[str, str]] = Field(default_factory=lambda: [
        {"name": "Eurostat International Trade in Goods",
         "url": "https://ec.europa.eu/eurostat/web/international-trade-in-goods",
         "description": "Provides statistical data on international trade in goods for the EU."},
        {"name": "Fastmarkets", "url": "https://www.fastmarkets.com/",
         "description": "Offers market intelligence on global commodity prices and trends."},
        {"name": "Trade Data Monitor", "url": "https://tradedatamonitor.com/",
         "description": "Aggregates trade data from multiple countries to monitor global trade flows."},
        {"name": "USDA ESRQuery", "url": "https://apps.fas.usda.gov/esrquery/ESRHome.aspx",
         "description": "Delivers export sales reporting data from the USDA."},
        {"name": "USDA Foreign Agricultural Service", "url": "https://www.fas.usda.gov/",
         "description": "Focuses on international trade policy and export support for U.S. agriculture."},
        {"name": "USDA National Agricultural Statistics Service", "url": "https://www.nass.usda.gov/",
         "description": "Provides comprehensive agricultural statistics for the U.S."}
    ])

    def _run(self, query: str) -> Any:
        """
        Verilen sorgu için en uygun veri kaynağını seçer
        """
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

    async def _arun(self, query: str) -> Any:
        """
        Asenkron çalışma metodu - şu an için senkron metodu kullanıyoruz
        """
        return self._run(query) 