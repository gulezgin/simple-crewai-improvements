import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()


class OpenAIConfig:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_API_BASE")
        self.subscription_key = os.getenv("AZURE_API_KEY")
        self.api_version = os.getenv("AZURE_API_VERSION")
        self.deployment = os.getenv("AZURE_API_MODEL")


openai_config = OpenAIConfig()
