"""Factory: elige el cliente concreto según la configuración. El resto del código no se entera de cuál es."""
import os
from collections.abc import AsyncIterator

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from schemas import ChatMessage, LLMConfig, ModelResponse, Provider


class AsyncLLMManager:
    # Registro Provider -> clase. Sumar un proveedor nuevo es agregar una línea acá.
    _registry: dict[Provider, type[BaseLLMClient]] = {
        Provider.OPENAI: OpenAIClient,
        Provider.ANTHROPIC: AnthropicClient,
    }

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._create_client(config)

    @classmethod
    def from_env(cls, provider: Provider | str | None = None) -> "AsyncLLMManager":
        """Usa el proveedor pedido o, si no se pasa ninguno, el de LLM_PROVIDER (por defecto openai)."""
        raw = provider or os.getenv("LLM_PROVIDER") or Provider.OPENAI.value
        try:
            chosen = raw if isinstance(raw, Provider) else Provider(raw.strip().lower())
        except ValueError:
            opciones = ", ".join(p.value for p in Provider)
            raise ValueError(f"LLM_PROVIDER inválido: {raw!r}. Opciones: {opciones}") from None
        return cls(LLMConfig.from_env(chosen))

    @classmethod
    def _create_client(cls, config: LLMConfig) -> BaseLLMClient:
        client_cls = cls._registry.get(config.provider)
        if client_cls is None:
            raise ValueError(f"Proveedor no soportado: {config.provider}")
        return client_cls(config)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async for token in self.client.generate_stream(messages):
            yield token
