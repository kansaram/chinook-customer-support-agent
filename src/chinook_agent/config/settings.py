import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings:
    # App Config
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Models
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4o")
    
    # Paths
    DB_FILE_PATH: Path = BASE_DIR / os.getenv("DB_FILE_PATH", "data/chinook.db")
    MEMORY_DB_PATH: Path = BASE_DIR / os.getenv("MEMORY_DB_PATH", "data/customer_memory.db")
    CHINOOK_SQL_URL: str = os.getenv(
        "CHINOOK_SQL_URL", 
        "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)

settings = Settings()