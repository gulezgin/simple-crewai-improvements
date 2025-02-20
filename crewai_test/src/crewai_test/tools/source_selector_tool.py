from langchain.tools import BaseTool
from typing import Optional
from ..config.config import openai_config
import openai

class SourceSelectorTool(BaseTool):
    name = "Tarım Veri Kaynağı Seçici"
    description = "Tarım ile ilgili veri kaynaklarını seçmek ve değerlendirmek için kullanılan araç"

    def __init__(self):
        super().__init__()
        self.sources = [
            {"name": "Eurostat International Trade in Goods",
             "url": "https://ec.europa.eu/eurostat/web/international-trade-in-goods",
             "description": "AB için uluslararası mal ticareti istatistiksel verileri."},
            {"name": "Fastmarkets",
             "url": "https://www.fastmarkets.com/",
             "description": "Küresel emtia fiyatları ve trendleri hakkında pazar istihbaratı."},
            {"name": "USDA Foreign Agricultural Service",
             "url": "https://www.fas.usda.gov/",
             "description": "ABD tarımı için uluslararası ticaret politikası ve ihracat desteği."},
            {"name": "Trade Data Monitor", "url": "https://tradedatamonitor.com/",
             "description": "Aggregates trade data from multiple countries to monitor global trade flows."},
            {"name": "USDA ESRQuery", "url": "https://apps.fas.usda.gov/esrquery/ESRHome.aspx",
             "description": "Delivers export sales reporting data from the USDA."},
            {"name": "USDA National Agricultural Statistics Service", "url": "https://www.nass.usda.gov/",
             "description": "Provides comprehensive agricultural statistics for the U.S."}
        ]

    def _run(self, query: str) -> dict:
        prompt = (
            "Aşağıdaki veri kaynakları listesinden, verilen sorgu için en uygun olanı seçin:\n"
            + "\n".join([f"{s['name']}: {s['description']}" for s in self.sources])
            + f"\n\nKullanıcı sorgusu: {query}\n"
            + "Hangi veri kaynağı en alakalı? Sadece kaynak adı ve URL'sini yanıtlayın."
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
        return {"name": "Bulunamadı", "url": "Eşleşen kaynak bulunamadı"}

    def _arun(self, query: str) -> dict:
        raise NotImplementedError("Bu araç asenkron çalışmayı desteklemiyor") 