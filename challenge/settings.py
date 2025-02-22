from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    OPENROUTER_API_KEY: str
    HUGGINFACE_API_KEY: str
    MODELS: str
    QUESTION_PROMPT: str
    RANKING_PROMPT: str
