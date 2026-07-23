"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings sourced from the environment / a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database -----------------------------------------------------------
    # NOTE: This must be a DIRECT (non-SRV) connection string. The application
    # intentionally rejects mongodb+srv:// URIs (see the validator below).
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017/?directConnection=true",
        alias="MONGODB_URI",
    )
    mongodb_db: str = Field(default="factorypilot", alias="MONGODB_DB")

    # --- Multi-tenancy ------------------------------------------------------
    # Tenant used when a request carries no X-Tenant-Id header. Set it to ""
    # to make the header mandatory (every unheadered request then 400s).
    default_tenant_id: str = Field(default="demo", alias="DEFAULT_TENANT_ID")

    # --- API metadata -------------------------------------------------------
    api_title: str = Field(default="FactoryPilot AI", alias="API_TITLE")
    api_version: str = Field(default="0.1.0", alias="API_VERSION")

    # --- LLM ------------------------------------------------------------------
    # Model used for every Anthropic API call in this codebase: RCA narrative
    # synthesis, orchestrator routing, and narrative composition. One setting,
    # so a model change is a one-line edit rather than a grep across services.
    # Every call site is optional and falls back to deterministic behaviour
    # when ANTHROPIC_API_KEY is unset, regardless of this value.
    anthropic_model: str = Field(default="claude-opus-4-8", alias="ANTHROPIC_MODEL")
    # Empty means "no LLM configured" — every call site already treats that as
    # optional and falls back to deterministic behaviour, never an error.
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # --- CORS ---------------------------------------------------------------
    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")

    # --- Sensor pipeline ----------------------------------------------------
    # Which source feeds the pipeline: simulator | opcua | mqtt. Nothing
    # downstream branches on this — it only decides what gets constructed.
    sensor_source: str = Field(default="simulator", alias="SENSOR_SOURCE")
    sensor_interval_seconds: float = Field(
        default=2.0, alias="SENSOR_INTERVAL_SECONDS", gt=0
    )
    # Simulated seconds per wall-clock second. 15120 replays three weeks of
    # degradation in two minutes for a demo, without altering the model.
    sensor_time_scale: float = Field(default=1.0, alias="SENSOR_TIME_SCALE", gt=0)

    # --- Ingestion ----------------------------------------------------------
    ingestion_enabled: bool = Field(default=True, alias="INGESTION_ENABLED")
    ingestion_batch_size: int = Field(default=200, alias="INGESTION_BATCH_SIZE", gt=0)
    ingestion_flush_seconds: float = Field(
        default=5.0, alias="INGESTION_FLUSH_SECONDS", gt=0
    )
    readings_retention_days: int = Field(
        default=90, alias="READINGS_RETENTION_DAYS", ge=0
    )

    # --- Knowledge base / RAG ------------------------------------------------
    # local | api | hashing. 'local' runs sentence-transformers in-process;
    # 'hashing' needs no model download and is what the tests use.
    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    # Only consulted for providers that cannot report their own width (api,
    # hashing). The local provider reports the loaded model's true dimension.
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION", gt=0)
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE", gt=0)
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_api_endpoint: str = Field(default="", alias="EMBEDDING_API_ENDPOINT")

    # auto | atlas | numpy. 'auto' probes the cluster and falls back.
    vector_backend: str = Field(default="auto", alias="VECTOR_BACKEND")
    vector_index_name: str = Field(
        default="chunk_vector_index", alias="VECTOR_INDEX_NAME"
    )
    vector_index_path: str = Field(
        default="ml/artifacts/vector_index.jsonl", alias="VECTOR_INDEX_PATH"
    )

    # Retrieval tuning.
    rag_chunk_target_tokens: int = Field(default=500, alias="RAG_CHUNK_TARGET_TOKENS", gt=0)
    rag_chunk_overlap_tokens: int = Field(default=80, alias="RAG_CHUNK_OVERLAP_TOKENS", ge=0)
    rag_top_k: int = Field(default=6, alias="RAG_TOP_K", gt=0)
    # Below this blended score a passage is not worth showing; the retriever
    # returns an empty result with a reason rather than weak matches.
    # Scores are raw cosine blended with BM25, so this is a real threshold:
    # unrelated text lands near 0.1, a genuine match near 0.5+.
    rag_min_score: float = Field(default=0.25, alias="RAG_MIN_SCORE", ge=0.0, le=1.0)
    # Weight of the vector half of the hybrid score; the keyword half gets the
    # remainder. Error codes and part numbers are exact strings that embeddings
    # match poorly, so the keyword half is far from negligible.
    rag_vector_weight: float = Field(
        default=0.6, alias="RAG_VECTOR_WEIGHT", ge=0.0, le=1.0
    )
    rag_max_upload_bytes: int = Field(
        default=25 * 1024 * 1024, alias="RAG_MAX_UPLOAD_BYTES", gt=0
    )

    # --- Orchestrator --------------------------------------------------------
    # Per-module ceiling. A module that exceeds it is reported UNAVAILABLE and
    # the request continues without it — never a failed request.
    orchestrator_module_timeout_seconds: float = Field(
        default=15.0, alias="ORCHESTRATOR_MODULE_TIMEOUT_SECONDS", gt=0
    )
    # How long a machine's module results stay reusable for follow-up turns in
    # the same conversation. Short: sensor state moves.
    orchestrator_cache_ttl_seconds: float = Field(
        default=120.0, alias="ORCHESTRATOR_CACHE_TTL_SECONDS", ge=0
    )
    # The routing LLM only picks modules, so it is on the critical path of
    # every request and gets a tighter budget than the modules themselves.
    orchestrator_router_timeout_seconds: float = Field(
        default=8.0, alias="ORCHESTRATOR_ROUTER_TIMEOUT_SECONDS", gt=0
    )

    # --- OPC UA (used only when SENSOR_SOURCE=opcua) -------------------------
    opcua_endpoint: str = Field(
        default="opc.tcp://localhost:4840/factorypilot", alias="OPCUA_ENDPOINT"
    )
    opcua_node_map_path: str = Field(
        default="config/opcua_nodes.json", alias="OPCUA_NODE_MAP_PATH"
    )
    opcua_username: str = Field(default="", alias="OPCUA_USERNAME")
    opcua_password: str = Field(default="", alias="OPCUA_PASSWORD")
    opcua_security_policy: str = Field(default="None", alias="OPCUA_SECURITY_POLICY")
    opcua_publishing_interval_ms: int = Field(
        default=1000, alias="OPCUA_PUBLISHING_INTERVAL_MS", gt=0
    )

    # --- MQTT (used only when SENSOR_SOURCE=mqtt) ---------------------------
    mqtt_host: str = Field(default="localhost", alias="MQTT_HOST")
    mqtt_port: int = Field(default=1883, alias="MQTT_PORT")
    mqtt_topic_map_path: str = Field(
        default="config/mqtt_topics.json", alias="MQTT_TOPIC_MAP_PATH"
    )
    mqtt_username: str = Field(default="", alias="MQTT_USERNAME")
    mqtt_password: str = Field(default="", alias="MQTT_PASSWORD")
    mqtt_client_id: str = Field(default="factorypilot-ingest", alias="MQTT_CLIENT_ID")
    mqtt_qos: int = Field(default=1, alias="MQTT_QOS", ge=0, le=2)
    mqtt_tls: bool = Field(default=False, alias="MQTT_TLS")

    @field_validator("mongodb_uri")
    @classmethod
    def _reject_srv(cls, value: str) -> str:
        if value.strip().lower().startswith("mongodb+srv://"):
            raise ValueError(
                "MONGODB_URI must be a direct (non-SRV) connection string. "
                "mongodb+srv:// URIs are not supported — list the hosts directly "
                "with mongodb://host1,host2,host3/..."
            )
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a clean list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
