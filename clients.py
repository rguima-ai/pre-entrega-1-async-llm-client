"""Clientes de LLM: un contrato común (ABC) y una implementación por proveedor.

La política de reintentos, timeouts y logging vive una sola vez, en BaseLLMClient.
Cada proveedor solo implementa cómo hablarle a su SDK (_complete y _stream).
"""
import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import anthropic
import openai

from schemas import ChatMessage, LLMConfig, ModelResponse, Provider, Role

log = logging.getLogger("llm")

# Prefijo del último chunk que emite generate_stream() cuando falla, para que el consumidor lo detecte.
STREAM_ERROR_PREFIX = "\n[ERROR] "


def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


class BaseLLMClient(ABC):
    """Contrato común: el resto del código solo conoce generate() y generate_stream()."""

    provider: Provider
    # Errores TRANSITORIOS: tiene sentido reintentar. Cada subclase suma los de su SDK.
    # TimeoutError es el que lanza asyncio.timeout() cuando una llamada se cuelga.
    transient_errors: tuple[type[Exception], ...] = (TimeoutError,)
    # Errores esperables de la API. Cualquier otro es un bug nuestro y se loguea con traceback.
    api_errors: tuple[type[Exception], ...] = (TimeoutError,)

    def __init__(self, config: LLMConfig):
        if config.provider != self.provider:
            raise ValueError(
                f"{type(self).__name__} necesita una config de {self.provider.value}, "
                f"recibió una de {config.provider.value}"
            )
        self.config = config
        self.model = config.model

    # ---------- lo que implementa cada proveedor ----------
    @abstractmethod
    async def _complete(self, messages: list[ChatMessage]) -> str:
        """Una llamada al SDK en modo normal. Devuelve el texto de la respuesta."""

    @abstractmethod
    def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Una llamada al SDK en modo streaming, como generador asíncrono de tokens."""

    # ---------- API pública (igual para todos los proveedores) ----------
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Respuesta completa. Nunca lanza: si algo falla, devuelve un ModelResponse con error."""
        t0 = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                # Timeout por intento: si la API se cuelga, no nos colgamos con ella.
                async with asyncio.timeout(self.config.timeout_s):
                    text = await self._complete(messages)
            except Exception as e:
                if self._should_retry(e, attempt):
                    await self._wait_before_retry(e, attempt)
                    continue
                self._log_failure(e, attempt)
                return self._error_response(f"{type(e).__name__}: {e}", attempt, t0)

            if not text.strip():
                log.error("vacia provider=%s intentos=%d", self.provider.value, attempt)
                return self._error_response("El modelo devolvió una respuesta vacía", attempt, t0)

            latency_ms = _ms_since(t0)
            log.info("ok provider=%s model=%s intentos=%d latencia_ms=%.0f",
                     self.provider.value, self.model, attempt, latency_ms)
            return ModelResponse(provider=self.provider, model=self.model, content=text,
                                 attempts=attempt, latency_ms=latency_ms)

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Respuesta token a token. Nunca lanza: si falla, el último chunk empieza con STREAM_ERROR_PREFIX.

        Solo se reintenta si el error llega ANTES del primer token. Si el usuario ya vio
        medio texto, reintentar lo duplicaría: ahí se corta y se informa el error.
        """
        t0 = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            tokens = self._stream(messages)
            started = False
            try:
                while True:
                    # Timeout por chunk: corta si el primer token (o el siguiente) no llega a tiempo.
                    async with asyncio.timeout(self.config.timeout_s):
                        token = await anext(tokens, None)
                    if token is None:
                        break
                    if not started:
                        started = True
                        log.info("ttft provider=%s intento=%d ttft_ms=%.0f",
                                 self.provider.value, attempt, _ms_since(t0))
                    yield token
            except Exception as e:
                if not started and self._should_retry(e, attempt):
                    await self._wait_before_retry(e, attempt)
                    continue
                self._log_failure(e, attempt)
                yield f"{STREAM_ERROR_PREFIX}{type(e).__name__}: {e}"
                return
            finally:
                await tokens.aclose()  # cierra la conexión HTTP aunque cortemos a mitad

            log.info("stream_ok provider=%s intentos=%d total_ms=%.0f",
                     self.provider.value, attempt, _ms_since(t0))
            return

    # ---------- resiliencia ----------
    def _should_retry(self, error: Exception, attempt: int) -> bool:
        # Los permanentes (401, 400, 404...) van a fallar igual: no se reintentan.
        return isinstance(error, self.transient_errors) and attempt <= self.config.max_retries

    def _backoff_delay(self, attempt: int) -> float:
        # Exponencial (0.5, 1, 2, 4...) + jitter, para que muchos clientes no reintenten en el mismo instante.
        base = self.config.base_delay_s
        return base * 2 ** (attempt - 1) + random.uniform(0, base)

    async def _wait_before_retry(self, error: Exception, attempt: int) -> None:
        delay = self._backoff_delay(attempt)
        log.warning("reintento provider=%s intento=%d error=%s espera_s=%.2f",
                    self.provider.value, attempt, type(error).__name__, delay)
        await asyncio.sleep(delay)  # asyncio.sleep, no time.sleep: el event loop sigue libre

    def _log_failure(self, error: Exception, attempt: int) -> None:
        kind = "transitorio_agotado" if isinstance(error, self.transient_errors) else "permanente"
        unexpected = not isinstance(error, self.api_errors)
        log.error("fallo provider=%s tipo=%s intentos=%d error=%s",
                  self.provider.value, kind, attempt, type(error).__name__,
                  exc_info=error if unexpected else None)

    def _error_response(self, error: str, attempt: int, t0: float) -> ModelResponse:
        return ModelResponse(provider=self.provider, model=self.model, error=error,
                             attempts=attempt, latency_ms=_ms_since(t0))


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI
    # APIConnectionError incluye APITimeoutError; InternalServerError cubre los 5xx.
    transient_errors = (TimeoutError, openai.RateLimitError, openai.APIConnectionError,
                        openai.InternalServerError)
    api_errors = (TimeoutError, openai.APIError)

    def __init__(self, config: LLMConfig, sdk_client: openai.AsyncOpenAI | None = None):
        super().__init__(config)
        # max_retries=0: la política de reintentos es nuestra. Si no, el SDK reintenta
        # 2 veces más por su cuenta y los reintentos se multiplican sin que nos enteremos.
        self._sdk = sdk_client or openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            timeout=config.timeout_s,
            max_retries=0,
        )

    def _params(self, messages: list[ChatMessage]) -> dict:
        # OpenAI acepta el rol "system" como un mensaje más.
        return {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    async def _complete(self, messages: list[ChatMessage]) -> str:
        response = await self._sdk.chat.completions.create(**self._params(messages))
        return response.choices[0].message.content or ""

    async def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._sdk.chat.completions.create(**self._params(messages), stream=True)
        async with stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC
    # OverloadedError (529, "API sobrecargada"), ServiceUnavailableError y DeadlineExceededError
    # NO heredan de InternalServerError en este SDK: hay que listarlos aparte.
    transient_errors = (TimeoutError, anthropic.RateLimitError, anthropic.APIConnectionError,
                        anthropic.InternalServerError, anthropic.OverloadedError,
                        anthropic.ServiceUnavailableError, anthropic.DeadlineExceededError)
    api_errors = (TimeoutError, anthropic.APIError)

    def __init__(self, config: LLMConfig, sdk_client: anthropic.AsyncAnthropic | None = None):
        super().__init__(config)
        self._sdk = sdk_client or anthropic.AsyncAnthropic(
            api_key=config.api_key.get_secret_value(),
            timeout=config.timeout_s,
            max_retries=0,  # mismo motivo que en OpenAIClient
        )

    def _params(self, messages: list[ChatMessage]) -> dict:
        # Anthropic no acepta role="system" dentro de messages: va en el parámetro `system`.
        system = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        params = {
            "model": self.model,
            "max_tokens": self.config.max_tokens,  # obligatorio en Anthropic
            "messages": [{"role": m.role.value, "content": m.content}
                         for m in messages if m.role != Role.SYSTEM],
            # Sin temperature a propósito: el SDK actual de Anthropic (>= 1.0) da TypeError si se la pasamos.
        }
        if system:
            params["system"] = system
        return params

    async def _complete(self, messages: list[ChatMessage]) -> str:
        response = await self._sdk.messages.create(**self._params(messages))
        # content es una lista de bloques; nos quedamos con los de texto.
        return "".join(block.text for block in response.content if block.type == "text")

    async def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async with self._sdk.messages.stream(**self._params(messages)) as stream:
            async for text in stream.text_stream:
                yield text
