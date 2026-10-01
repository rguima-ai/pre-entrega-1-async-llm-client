"""Contratos de datos: qué entra (mensajes), cómo se configura el modelo y qué sale."""
import os
from enum import Enum

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


# Defaults vigentes. Se pisan con OPENAI_MODEL / ANTHROPIC_MODEL en el .env,
# así un modelo retirado se cambia sin tocar código.
DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-haiku-4-5-20251001",
}
API_KEY_ENV = {Provider.OPENAI: "OPENAI_API_KEY", Provider.ANTHROPIC: "ANTHROPIC_API_KEY"}
MODEL_ENV = {Provider.OPENAI: "OPENAI_MODEL", Provider.ANTHROPIC: "ANTHROPIC_MODEL"}
# Parámetros opcionales comunes a ambos proveedores: campo de LLMConfig -> variable de entorno
OPTIONAL_ENV = {
    "temperature": "LLM_TEMPERATURE",
    "max_tokens": "LLM_MAX_TOKENS",
    "timeout_s": "LLM_TIMEOUT_S",
    "max_retries": "LLM_MAX_RETRIES",
}


class LLMConfig(BaseModel):
    """Configuración validada: si algo está mal, falla al arrancar y no en la llamada número 500."""

    provider: Provider
    api_key: SecretStr  # se imprime como '**********'; la key real solo sale con get_secret_value()
    model: str | None = None
    # Anthropic no la recibe (ver AnthropicClient), pero la validamos igual: es parte del contrato.
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay_s: float = Field(default=0.5, gt=0)

    @field_validator("api_key")
    @classmethod
    def key_no_vacia(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().strip():
            raise ValueError("La API key está vacía")
        return v

    @model_validator(mode="after")
    def modelo_por_defecto(self) -> "LLMConfig":
        if not self.model:
            self.model = DEFAULT_MODELS[self.provider]
        return self

    @classmethod
    def from_env(cls, provider: Provider) -> "LLMConfig":
        """Arma la config de un proveedor leyendo variables de entorno (nunca keys hardcodeadas)."""
        key_var = API_KEY_ENV[provider]
        api_key = os.getenv(key_var)
        if not api_key:
            raise ValueError(f"Falta {key_var} en el entorno (.env) para usar {provider.value}")
        optional = {field: os.environ[var] for field, var in OPTIONAL_ENV.items() if os.getenv(var)}
        return cls(provider=provider, api_key=api_key, model=os.getenv(MODEL_ENV[provider]), **optional)


class ModelResponse(BaseModel):
    """Salida uniforme de cualquier proveedor. Si falló, `error` explica por qué."""

    provider: Provider
    model: str
    content: str = ""
    error: str | None = None
    # Solo cuando fallaron principal y fallback: el error de cada uno, en el orden en que se probaron.
    # En ese caso `provider` es el último que se intentó, no uno que haya respondido.
    provider_errors: dict[Provider, str] = Field(default_factory=dict)
    attempts: int = Field(default=1, ge=1)
    latency_ms: float = Field(default=0.0, ge=0)

    @property
    def ok(self) -> bool:
        return self.error is None

    @model_validator(mode="after")
    def contenido_o_error(self) -> "ModelResponse":
        # Una respuesta vacía sin error no puede pasar por buena.
        if self.error is None and not self.content.strip():
            raise ValueError("Una respuesta sin error tiene que traer contenido")
        return self
