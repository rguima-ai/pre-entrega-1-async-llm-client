"""Factory + fallback: elige el cliente según la configuración y, si falla, conmuta al de respaldo.

El resto del código no se entera de qué proveedor respondió.
"""
import logging
import os
from collections.abc import AsyncIterator

from clients import STREAM_ERROR_PREFIX, AnthropicClient, BaseLLMClient, OpenAIClient
from schemas import ChatMessage, LLMConfig, ModelResponse, Provider

log = logging.getLogger("llm")


def _parse_provider(raw: Provider | str, var_name: str) -> Provider:
    if isinstance(raw, Provider):
        return raw
    try:
        return Provider(raw.strip().lower())
    except ValueError:
        opciones = ", ".join(p.value for p in Provider)
        raise ValueError(f"{var_name} inválido: {raw!r}. Opciones: {opciones}") from None


class AsyncLLMManager:
    # Registro Provider -> clase. Sumar un proveedor nuevo es agregar una línea acá.
    _registry: dict[Provider, type[BaseLLMClient]] = {
        Provider.OPENAI: OpenAIClient,
        Provider.ANTHROPIC: AnthropicClient,
    }

    def __init__(self, config: LLMConfig, fallback_config: LLMConfig | None = None):
        if fallback_config and fallback_config.provider == config.provider:
            raise ValueError(
                f"El fallback tiene que ser un proveedor distinto del principal "
                f"(los dos son {config.provider.value}). Revisá LLM_FALLBACK_PROVIDER."
            )
        self.config = config
        self.fallback_config = fallback_config
        self.client = self._create_client(config)
        self.fallback_client = self._create_client(fallback_config) if fallback_config else None

    @classmethod
    def from_env(cls, provider: Provider | str | None = None, *,
                 with_fallback: bool = True) -> "AsyncLLMManager":
        """Principal: el proveedor pedido o LLM_PROVIDER (por defecto openai).
        Respaldo: LLM_FALLBACK_PROVIDER, si está definido y with_fallback es True."""
        principal = _parse_provider(provider or os.getenv("LLM_PROVIDER") or Provider.OPENAI.value,
                                    "LLM_PROVIDER")
        fallback_raw = os.getenv("LLM_FALLBACK_PROVIDER") if with_fallback else None
        fallback_config = (LLMConfig.from_env(_parse_provider(fallback_raw, "LLM_FALLBACK_PROVIDER"))
                           if fallback_raw else None)
        return cls(LLMConfig.from_env(principal), fallback_config)

    @classmethod
    def _create_client(cls, config: LLMConfig) -> BaseLLMClient:
        client_cls = cls._registry.get(config.provider)
        if client_cls is None:
            raise ValueError(f"Proveedor no soportado: {config.provider}")
        return client_cls(config)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        # El principal ya hizo sus reintentos. Si igual falló (reintentos agotados o error
        # permanente, como una key inválida o un modelo retirado), probamos con el respaldo.
        response = await self.client.generate(messages)
        if response.ok or self.fallback_client is None:
            return response

        self._log_fallback(response.error)
        backup = await self.fallback_client.generate(messages)
        if backup.ok:
            return backup
        # Fallaron los dos: devolvemos ambos errores para saber qué pasó con cada uno.
        both = (f"{self.config.provider.value}: {response.error} | "
                f"{self.fallback_config.provider.value}: {backup.error}")
        return backup.model_copy(update={"error": both})

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        # Fallback solo si el principal falla ANTES del primer token: si el usuario ya vio
        # medio texto, cambiar de modelo a mitad de frase sería peor que informar el error.
        started = False
        async for token in self.client.generate_stream(messages):
            if not started and self.fallback_client and token.startswith(STREAM_ERROR_PREFIX):
                self._log_fallback(token.removeprefix(STREAM_ERROR_PREFIX))
                async for backup_token in self.fallback_client.generate_stream(messages):
                    yield backup_token
                return
            started = True
            yield token

    def _log_fallback(self, error: str | None) -> None:
        log.warning("fallback de=%s a=%s motivo=%s", self.config.provider.value,
                    self.fallback_config.provider.value, (error or "").split(":")[0])
