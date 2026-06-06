from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # PostgreSQL
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str

    # Triton
    triton_host: str = "localhost"
    triton_http_port: int = 8000
    triton_grpc_port: int = 8001

    # App
    app_env: str = "production"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def triton_http_url(self) -> str:
        return f"{self.triton_host}:{self.triton_http_port}"


settings = Settings()
