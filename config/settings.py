"""Application configuration management."""

import os
from pathlib import Path
from typing import Optional, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    """Application settings."""
    
    # Project
    PROJECT_NAME = "Bedrock LLM Comparison"
    VERSION = "1.0.0"
    
    # AWS
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    
    # Bedrock Regions (per model)
    BEDROCK_CLAUDE_REGION: str = os.getenv("BEDROCK_CLAUDE_REGION", "us-east-1")
    BEDROCK_TITAN_REGION: str = os.getenv("BEDROCK_TITAN_REGION", "us-east-1")
    BEDROCK_LLAMA_REGION: str = os.getenv("BEDROCK_LLAMA_REGION", "us-east-1")
    BEDROCK_DEEPSEEK_REGION: str = os.getenv("BEDROCK_DEEPSEEK_REGION", "us-west-2")
    BEDROCK_MISTRAL_REGION: str = os.getenv("BEDROCK_MISTRAL_REGION", "us-east-1")
    
    # Model ID Mapping
    MODEL_IDS: Dict[str, str] = {
        "claude": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "titan": "amazon.titan-text-express-v1",
        "llama": "meta.llama3-70b-instruct-v1:0",
        "deepseek": "us.deepseek.r1-v1:0",
        "mistral": "mistral.mistral-7b-instruct-v0:2"
    }
    
    # Region Mapping (for dynamic region lookup)
    MODEL_REGIONS: Dict[str, str] = {
        "claude": BEDROCK_CLAUDE_REGION,
        "titan": BEDROCK_TITAN_REGION,
        "llama": BEDROCK_LLAMA_REGION,
        "deepseek": BEDROCK_DEEPSEEK_REGION,
        "mistral": BEDROCK_MISTRAL_REGION
    }
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Database
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    # Paths
    DATA_DIR = BASE_DIR / "data"
    INPUT_DIR = DATA_DIR / "input"
    OUTPUT_DIR = DATA_DIR / "output" / "reports"
    CACHE_DIR = DATA_DIR / "cache"
    LOG_DIR = DATA_DIR / "output" / "logs"
    
    # Application
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "5"))
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    
    # Model Defaults
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
    
    # Model-Specific Parameters
    MISTRAL_TOP_K: int = int(os.getenv("MISTRAL_TOP_K", "50"))
    MISTRAL_TOP_P: float = float(os.getenv("MISTRAL_TOP_P", "0.9"))
    
    # Available Models
    BEDROCK_MODELS = ["claude", "titan", "llama", "deepseek", "mistral"]
    API_MODELS = ["openai", "gemini"]
    ALL_MODELS = BEDROCK_MODELS + API_MODELS
    
    @classmethod
    def get_model_id(cls, model_name: str) -> str:
        """Get the Bedrock model ID for a given model name."""
        return cls.MODEL_IDS.get(model_name, "")
    
    @classmethod
    def get_model_region(cls, model_name: str) -> str:
        """Get the AWS region for a given model."""
        return cls.MODEL_REGIONS.get(model_name, cls.AWS_DEFAULT_REGION)
    
    @classmethod
    def validate(cls):
        """Validate required settings."""
        required = []
        
        if not cls.AWS_ACCESS_KEY_ID:
            required.append("AWS_ACCESS_KEY_ID")
        if not cls.AWS_SECRET_ACCESS_KEY:
            required.append("AWS_SECRET_ACCESS_KEY")
        
        if required:
            raise ValueError(f"Missing required environment variables: {', '.join(required)}")
    
    @classmethod
    def ensure_directories(cls):
        """Create required directories if they don't exist."""
        for directory in [cls.DATA_DIR, cls.INPUT_DIR, cls.OUTPUT_DIR, 
                         cls.CACHE_DIR, cls.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
