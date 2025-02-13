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