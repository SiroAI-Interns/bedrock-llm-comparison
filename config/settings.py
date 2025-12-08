"""Application configuration management."""

import os
from pathlib import Path
from typing import Optional
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
    
    # Bedrock Regions
    BEDROCK_TITAN_REGION: str = os.getenv("BEDROCK_TITAN_REGION", "us-east-1")
    BEDROCK_LLAMA_REGION: str = os.getenv("BEDROCK_LLAMA_REGION", "us-west-2")
    BEDROCK_CLAUDE_REGION: str = os.getenv("BEDROCK_CLAUDE_REGION", "us-east-1")
    
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
