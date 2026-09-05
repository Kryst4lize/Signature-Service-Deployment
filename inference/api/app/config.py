from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_host: str
    postgres_port: int = 5432
    postgres_db: str
    postgres_user: str
    postgres_password: str

    # ── Triton ────────────────────────────────────────────────────────────────
    triton_host: str = "localhost"
    triton_http_port: int = 8000

    # Model names as they appear in triton/model_repository/.
    yolo_model: str = "yolov8s"
    denoiser_model: str = "latest_net_G_B"
    resnet_model: str = "resnet50_extractor"
    vgg_model: str = "vgg16_extractor"

    # ── Matching ──────────────────────────────────────────────────────────────
    # Cosine distance (0 = identical, 1 = orthogonal, 2 = opposite) averaged over
    # the two backbones. Calibrate against your own enrolled set: the training
    # pipeline's `sigtrain evaluate` prints the EER threshold as a cosine
    # *similarity* s, so set this to 1 - s.
    match_threshold: float = 0.30

    # YOLOv8 objectness below this is treated as "no signature on this page".
    detection_confidence: float = 0.5

    # ── Upload limits ─────────────────────────────────────────────────────────
    max_upload_bytes: int = 20 * 1024 * 1024
    max_pdf_pages: int = 20

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "production"

    # Comma-separated browser origins allowed to call this API.
    #
    # Deliberately typed `str`, not `list[str]`: pydantic-settings JSON-decodes
    # complex-typed fields in EnvSettingsSource *before* any validator runs, so
    # CORS_ORIGINS=http://localhost:8110 raises
    #   SettingsError: error parsing value for field "cors_origins"
    # at import time and crash-loops the container. Splitting in a property
    # keeps the env contract a plain comma-separated string.
    cors_origins: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def triton_http_url(self) -> str:
        return f"{self.triton_host}:{self.triton_http_port}"


# pydantic-settings fills the required fields from the environment, which a
# static checker cannot see.
settings = Settings()  # ty: ignore[missing-argument]
