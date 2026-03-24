from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    model: str = "gpt-4o-mini"
    batch_size: int = 10
    max_records: int = 1000
    data_dir: Path = Path("data")

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
