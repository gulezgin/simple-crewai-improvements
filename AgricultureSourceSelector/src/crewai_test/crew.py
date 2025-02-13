import os
import openai
from crewai_test.config.config import openai_config
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from dotenv import load_dotenv
from typing import Dict, Any
from .tools.source_selector_tool import SourceSelectorTool

# Uncomment the following line to use an example of a custom tool
# from crewai_test.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

# OpenAI yapılandırması
openai.api_type = "azure"
openai.api_base = openai_config.endpoint
openai.api_key = openai_config.subscription_key
openai.api_version = openai_config.api_version


class AgricultureSourceSelector:
    def __init__(self):
        self.sources = [
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


@CrewBase
class CrewaiTest():
    """Tarım Veri Analizi Crew'u"""

    def __init__(self):
        self.source_selector = SourceSelectorTool()
        super().__init__()

    def llm(self):
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path=env_path)

        return LLM(
            model=os.environ.get("AZURE_API_MODEL"),
            api_key=os.environ.get("AZURE_API_KEY"),
            base_url=os.environ.get("AZURE_API_BASE"),
            api_version=os.environ.get("AZURE_API_VERSION"),
        )

    @before_kickoff
    def setup_query(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Sorgu başlamadan önce gerekli hazırlıkları yap"""
        if 'query' not in inputs:
            inputs['query'] = "Tarım ürünleri ihracat verileri"
        inputs['analysis_depth'] = inputs.get('analysis_depth', 'detailed')
        return inputs

    @after_kickoff
    def log_results(self, output: Any) -> Any:
        """Sonuçları kaydet ve formatla"""
        print(f"Analiz Tamamlandı. Sonuçlar:")
        print("-" * 50)
        print(output)
        return output

    @agent
    def source_researcher(self) -> Agent:
        return Agent(
            name="Tarım Veri Kaynakları Uzmanı",
            role="Tarım sektörü veri kaynaklarını araştıran ve değerlendiren uzman",
            goal="Kullanıcının ihtiyacına en uygun, güvenilir ve güncel tarım veri kaynaklarını belirlemek",
            backstory="""
			15 yıllık deneyime sahip bir tarım veri analisti olarak, dünya çapındaki tüm önemli 
			tarım veri kaynaklarına hakimsiniz. Eurostat, USDA ve FAO gibi kurumların veri 
			sistemlerini derinlemesine biliyorsunuz. Veri kalitesi ve güncelliği konusunda 
			titizsiniz. Her zaman en doğru ve güncel kaynağı bulmak için çaba gösterirsiniz.
			""",
            tools=[self.source_selector],
            llm=self.llm(),
            verbose=True
        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            name="Tarım Veri Analisti",
            role="Tarım verilerini analiz eden ve yorumlayan uzman analist",
            goal="Veri kaynaklarından elde edilen bilgileri analiz ederek, anlamlı içgörüler çıkarmak",
            backstory="""
			Tarım ekonomisi alanında doktora derecesine sahip bir veri bilimcisiniz. 
			İstatistiksel analiz, veri madenciliği ve makine öğrenimi konularında uzmansınız. 
			Karmaşık tarım verilerini anlaşılır raporlara dönüştürme konusunda özel bir yeteneğiniz var. 
			Özellikle ticaret verileri, üretim tahminleri ve pazar analizi konularında deneyimlisiniz.
			""",
            llm=self.llm(),
            verbose=True
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            name="Tarım Rapor Uzmanı",
            role="Analiz sonuçlarını anlaşılır raporlara dönüştüren uzman",
            goal="Teknik analiz sonuçlarını herkesin anlayabileceği, açık ve net raporlar haline getirmek",
            backstory="""
			Tarım sektöründe 10 yıllık teknik yazarlık deneyimine sahipsiniz. 
			Karmaşık tarım verilerini ve analizleri, karar vericilerin ve çiftçilerin 
			anlayabileceği formatta sunma konusunda uzmansınız. Görselleştirme ve 
			veri hikayeleştirme konularında başarılı bir geçmişiniz var.
			""",
            llm=self.llm(),
            verbose=True
        )

    @task
    def find_source_task(self) -> Task:
        return Task(
            description="""
			1. Kullanıcının sorgusunu analiz et
			2. Sorguya en uygun veri kaynaklarını belirle
			3. Kaynakların güvenilirliğini ve güncelliğini kontrol et
			4. En uygun kaynağı seç ve seçim gerekçesini açıkla
			""",
            agent=self.source_researcher,
            context=lambda inputs: f"""
			Kullanıcı Sorgusu: {inputs['query']}
			Analiz Derinliği: {inputs['analysis_depth']}

			Lütfen bu sorgu için en uygun veri kaynağını belirleyin. 
			Kaynakın neden seçildiğini ve ne tür veriler içerdiğini açıklayın.
			""",
            expected_output="Seçilen kaynak, URL'si ve seçim gerekçesi"
        )

    @task
    def analyze_data_task(self) -> Task:
        return Task(
            description="""
			1. Seçilen kaynaktaki verileri analiz et
			2. Temel istatistikleri çıkar
			3. Trendleri belirle
			4. Önemli bulguları işaretle
			""",
            agent=self.data_analyst,
            context=lambda inputs: f"""
			Analiz Edilecek Kaynak: {inputs.get('selected_source', '')}
			Sorgu: {inputs['query']}

			Lütfen kaynaktaki verileri analiz edin ve önemli bulguları belirleyin.
			Varsa trend ve anomalileri işaretleyin.
			""",
            expected_output="Detaylı veri analizi raporu"
        )

    @task
    def create_report_task(self) -> Task:
        return Task(
            description="""
			1. Analiz sonuçlarını derle
			2. Anlaşılır bir format oluştur
			3. Önemli noktaları vurgula
			4. Görsel öğeler ekle
			""",
            agent=self.report_writer,
            context=lambda inputs: f"""
			Analiz Sonuçları: {inputs.get('analysis_results', '')}
			Hedef Kitle: Tarım sektörü profesyonelleri

			Lütfen analiz sonuçlarını anlaşılır bir rapora dönüştürün.
			Teknik detayları koruyarak ama anlaşılır bir dil kullanarak raporu hazırlayın.
			""",
            expected_output="Son kullanıcı raporu"
        )

    @crew
    def crew(self) -> Crew:
        """Tarım Veri Analizi Crew'unu oluştur"""
        return Crew(
            agents=[
                self.source_researcher(),
                self.data_analyst(),
                self.report_writer()
            ],
            tasks=[
                self.find_source_task(),
                self.analyze_data_task(),
                self.create_report_task()
            ],
            process=Process.sequential,
            verbose=True
        )
