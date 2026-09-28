"""Tests offline: los SDKs se reemplazan por objetos falsos. Sin internet ni API keys.

Los errores sí son los reales de openai/anthropic, así probamos la clasificación
transitorio/permanente tal como ocurriría en producción.
"""
import asyncio
from types import SimpleNamespace

import anthropic
import httpx2  # la librería HTTP que usan por debajo los SDKs actuales de openai y anthropic
import openai
import pytest
from pydantic import ValidationError

from clients import STREAM_ERROR_PREFIX, AnthropicClient, OpenAIClient
from manager import AsyncLLMManager
from schemas import ChatMessage, LLMConfig, ModelResponse, Provider, Role

PREGUNTA = [ChatMessage(role=Role.USER, content="¿Qué es la entropía?")]
ENV_VARS = ["LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_MODEL", "ANTHROPIC_MODEL",
            "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES"]


# ---------- errores reales de los SDKs ----------
def _request(sdk) -> httpx2.Request:
    url = ("https://api.openai.com/v1/chat/completions" if sdk is openai
           else "https://api.anthropic.com/v1/messages")
    return httpx2.Request("POST", url)


def _status_error(error_cls, sdk, status: int, message: str):
    return error_cls(message, response=httpx2.Response(status, request=_request(sdk)), body=None)


def rate_limit(sdk):
    return _status_error(sdk.RateLimitError, sdk, 429, "Rate limit")


def auth_error(sdk):
    return _status_error(sdk.AuthenticationError, sdk, 401, "Invalid API key")


def network_error(sdk):
    return sdk.APIConnectionError(request=_request(sdk))


# ---------- SDKs falsos ----------
class Slow:
    """Paso del guion que tarda `seconds` antes de resolverse (para probar timeouts)."""

    def __init__(self, seconds: float, value):
        self.seconds, self.value = seconds, value


async def _resolve(step):
    if isinstance(step, Slow):
        await asyncio.sleep(step.seconds)
        step = step.value
    if isinstance(step, Exception):
        raise step
    return step


async def _tokens(items):
    for item in items:
        if isinstance(item, Exception):  # falla a mitad del stream
            raise item
        yield item


class _FakeOpenAIStream:
    def __init__(self, items):
        self._items = items

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        async for text in _tokens(self._items):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class FakeOpenAI:
    """Imita AsyncOpenAI. Cada llamada consume el próximo paso del guion:
    un texto (modo normal), una lista de tokens (streaming), una excepción o un Slow."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        step = await _resolve(self.script.pop(0))
        if kwargs.get("stream"):
            return _FakeOpenAIStream(step)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=step))])


class _FakeAnthropicStream:
    def __init__(self, step):
        self._step = step

    async def __aenter__(self):
        # En el SDK real el request sale al entrar al context manager: ahí aparecen los errores.
        self.text_stream = _tokens(await _resolve(self._step))
        return self

    async def __aexit__(self, *exc):
        return False


class FakeAnthropic:
    """Imita AsyncAnthropic con el mismo sistema de guion que FakeOpenAI."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = await _resolve(self.script.pop(0))
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeAnthropicStream(self.script.pop(0))


# (clase del cliente, SDK falso, módulo del SDK real) para correr los mismos tests con ambos
PROVIDERS = [
    pytest.param(OpenAIClient, FakeOpenAI, openai, id="openai"),
    pytest.param(AnthropicClient, FakeAnthropic, anthropic, id="anthropic"),
]


def make_config(provider: Provider = Provider.OPENAI, **overrides) -> LLMConfig:
    # base_delay_s chiquito para que los reintentos no hagan lentos los tests
    params = {"provider": provider, "api_key": "sk-test", "timeout_s": 1.0,
              "max_retries": 2, "base_delay_s": 0.001}
    return LLMConfig(**(params | overrides))


def make_client(client_cls, fake, **overrides):
    return client_cls(make_config(client_cls.provider, **overrides), sdk_client=fake)


async def collect(stream) -> list[str]:
    return [token async for token in stream]


@pytest.fixture
def clean_env(monkeypatch):
    """Borra las variables del .env que pueda tener la máquina, para que el test controle todo."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# ---------- 1. Validación con Pydantic ----------
@pytest.mark.parametrize("temperature", [-0.1, 2.1, 5])
def test_temperature_fuera_de_rango_se_rechaza(temperature):
    with pytest.raises(ValidationError, match="temperature"):
        make_config(temperature=temperature)


@pytest.mark.parametrize("max_tokens", [0, -10])
def test_max_tokens_debe_ser_positivo(max_tokens):
    with pytest.raises(ValidationError, match="max_tokens"):
        make_config(max_tokens=max_tokens)


def test_api_key_vacia_se_rechaza():
    with pytest.raises(ValidationError, match="vacía"):
        make_config(api_key="   ")


def test_api_key_no_aparece_al_imprimir_la_config():
    config = make_config(api_key="sk-super-secreta")
    assert "sk-super-secreta" not in repr(config)
    assert "sk-super-secreta" not in str(config)
    assert config.api_key.get_secret_value() == "sk-super-secreta"


def test_rol_invalido_se_rechaza():
    with pytest.raises(ValidationError, match="role"):
        ChatMessage(role="robot", content="hola")


def test_mensaje_vacio_se_rechaza():
    with pytest.raises(ValidationError, match="content"):
        ChatMessage(role=Role.USER, content="")


@pytest.mark.parametrize("provider, modelo", [(Provider.OPENAI, "gpt-4o-mini"),
                                              (Provider.ANTHROPIC, "claude-haiku-4-5-20251001")])
def test_modelo_por_defecto_vigente(provider, modelo):
    assert make_config(provider).model == modelo


def test_respuesta_sin_error_exige_contenido():
    with pytest.raises(ValidationError, match="contenido"):
        ModelResponse(provider=Provider.OPENAI, model="x", content="")


# ---------- 2. Factory ----------
@pytest.mark.parametrize("valor, esperado", [("openai", OpenAIClient), ("anthropic", AnthropicClient),
                                             (" Anthropic ", AnthropicClient)])
def test_factory_elige_segun_llm_provider(clean_env, valor, esperado):
    clean_env.setenv("LLM_PROVIDER", valor)
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert isinstance(AsyncLLMManager.from_env().client, esperado)


def test_factory_usa_openai_si_no_hay_llm_provider(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    assert isinstance(AsyncLLMManager.from_env().client, OpenAIClient)


def test_factory_lee_modelo_y_parametros_del_env(clean_env):
    clean_env.setenv("LLM_PROVIDER", "anthropic")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    clean_env.setenv("ANTHROPIC_MODEL", "claude-otro-modelo")
    clean_env.setenv("LLM_MAX_TOKENS", "256")
    manager = AsyncLLMManager.from_env()
    assert manager.client.model == "claude-otro-modelo"
    assert manager.config.max_tokens == 256


def test_factory_rechaza_proveedor_desconocido(clean_env):
    clean_env.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="LLM_PROVIDER inválido"):
        AsyncLLMManager.from_env()


def test_factory_avisa_que_key_falta(clean_env):
    clean_env.setenv("LLM_PROVIDER", "anthropic")
    clean_env.setenv("OPENAI_API_KEY", "sk-test")  # la de otro proveedor no sirve
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AsyncLLMManager.from_env()


# ---------- 3. Modo normal y traducción a cada SDK ----------
@pytest.mark.parametrize("client_cls, fake_cls, sdk", PROVIDERS)
async def test_generate_devuelve_respuesta_validada(client_cls, fake_cls, sdk):
    client = make_client(client_cls, fake_cls("La entropía mide el desorden."))
    r = await client.generate(PREGUNTA)
    assert r.ok
    assert r.content == "La entropía mide el desorden."
    assert r.provider == client_cls.provider
    assert r.attempts == 1


async def test_openai_recibe_system_como_mensaje_y_temperature():
    fake = FakeOpenAI("ok")
    client = make_client(OpenAIClient, fake, temperature=0.3)
    await client.generate([ChatMessage(role=Role.SYSTEM, content="Sé breve."), *PREGUNTA])
    call = fake.calls[0]
    assert call["messages"][0] == {"role": "system", "content": "Sé breve."}
    assert call["temperature"] == 0.3


async def test_anthropic_separa_system_y_no_manda_temperature():
    fake = FakeAnthropic("ok")
    client = make_client(AnthropicClient, fake)
    await client.generate([ChatMessage(role=Role.SYSTEM, content="Sé breve."), *PREGUNTA])
    call = fake.calls[0]
    assert call["system"] == "Sé breve."
    assert all(m["role"] != "system" for m in call["messages"])
    assert "temperature" not in call
    assert call["max_tokens"] == 1024


async def test_respuesta_vacia_es_error_controlado():
    r = await make_client(OpenAIClient, FakeOpenAI("   ")).generate(PREGUNTA)
    assert not r.ok
    assert "vacía" in r.error


# ---------- 4. Resiliencia ----------
@pytest.mark.parametrize("client_cls, fake_cls, sdk", PROVIDERS)
async def test_retry_se_recupera_de_un_rate_limit(client_cls, fake_cls, sdk):
    fake = fake_cls(rate_limit(sdk), "respuesta después del 429")
    r = await make_client(client_cls, fake).generate(PREGUNTA)
    assert r.ok
    assert r.content == "respuesta después del 429"
    assert r.attempts == 2
    assert len(fake.calls) == 2


@pytest.mark.parametrize("client_cls, fake_cls, sdk", PROVIDERS)
async def test_rate_limit_persistente_devuelve_error_sin_crashear(client_cls, fake_cls, sdk):
    fake = fake_cls(*[rate_limit(sdk) for _ in range(3)])
    r = await make_client(client_cls, fake, max_retries=2).generate(PREGUNTA)
    assert not r.ok
    assert "RateLimitError" in r.error
    assert len(fake.calls) == 3  # 1 intento + 2 reintentos, ni uno más
    assert r.attempts == 3


@pytest.mark.parametrize("client_cls, fake_cls, sdk", PROVIDERS)
async def test_key_invalida_no_se_reintenta(client_cls, fake_cls, sdk):
    fake = fake_cls(auth_error(sdk), "no debería llegar acá")
    r = await make_client(client_cls, fake).generate(PREGUNTA)
    assert not r.ok
    assert "AuthenticationError" in r.error
    assert len(fake.calls) == 1  # error permanente: una sola llamada


@pytest.mark.parametrize("client_cls, fake_cls, sdk", PROVIDERS)
async def test_error_de_red_se_reintenta(client_cls, fake_cls, sdk):
    fake = fake_cls(network_error(sdk), network_error(sdk), "volvió la red")
    r = await make_client(client_cls, fake).generate(PREGUNTA)
    assert r.ok
    assert r.attempts == 3


async def test_anthropic_sobrecargado_529_se_reintenta():
    fake = FakeAnthropic(_status_error(anthropic.OverloadedError, anthropic, 529, "Overloaded"), "ok")
    r = await make_client(AnthropicClient, fake).generate(PREGUNTA)
    assert r.ok
    assert r.attempts == 2


async def test_error_de_red_persistente_devuelve_error_controlado():
    fake = FakeOpenAI(*[network_error(openai) for _ in range(3)])
    r = await make_client(OpenAIClient, fake, max_retries=2).generate(PREGUNTA)
    assert not r.ok
    assert "APIConnectionError" in r.error


async def test_timeout_se_trata_como_transitorio():
    fake = FakeOpenAI(Slow(5, "demasiado tarde"), "a tiempo")
    r = await make_client(OpenAIClient, fake, timeout_s=0.05).generate(PREGUNTA)
    assert r.ok
    assert r.content == "a tiempo"
    assert r.attempts == 2


def test_backoff_es_exponencial():
    client = make_client(OpenAIClient, FakeOpenAI(), base_delay_s=0.5)
    for attempt, esperado in [(1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0)]:
        delay = client._backoff_delay(attempt)
        assert esperado <= delay <= esperado + 0.5  # + jitter de hasta base_delay_s


# ---------- 5. Streaming ----------
@pytest.mark.parametrize("client_cls, fake_cls, sdk", PROVIDERS)
async def test_stream_devuelve_los_tokens_en_orden(client_cls, fake_cls, sdk):
    tokens = ["La ", "entropía ", "mide ", "el desorden."]
    client = make_client(client_cls, fake_cls(tokens))
    assert await collect(client.generate_stream(PREGUNTA)) == tokens


async def test_stream_es_un_generador_asincrono():
    stream = make_client(OpenAIClient, FakeOpenAI(["a"])).generate_stream(PREGUNTA)
    assert hasattr(stream, "__anext__")
    assert await collect(stream) == ["a"]


@pytest.mark.parametrize("client_cls, fake_cls, sdk", PROVIDERS)
async def test_stream_reintenta_si_falla_antes_del_primer_token(client_cls, fake_cls, sdk):
    fake = fake_cls(rate_limit(sdk), ["hola ", "mundo"])
    tokens = await collect(make_client(client_cls, fake).generate_stream(PREGUNTA))
    assert tokens == ["hola ", "mundo"]
    assert len(fake.calls) == 2


@pytest.mark.parametrize("client_cls, fake_cls, sdk", PROVIDERS)
async def test_stream_no_reintenta_despues_del_primer_token(client_cls, fake_cls, sdk):
    fake = fake_cls(["hola ", network_error(sdk)], ["no debería ", "repetirse"])
    tokens = await collect(make_client(client_cls, fake).generate_stream(PREGUNTA))
    assert tokens[0] == "hola "
    assert tokens[-1].startswith(STREAM_ERROR_PREFIX)
    assert "APIConnectionError" in tokens[-1]
    assert len(fake.calls) == 1


async def test_stream_con_key_invalida_devuelve_error_controlado():
    fake = FakeOpenAI(auth_error(openai))
    tokens = await collect(make_client(OpenAIClient, fake).generate_stream(PREGUNTA))
    assert len(tokens) == 1
    assert tokens[0].startswith(STREAM_ERROR_PREFIX)
    assert "AuthenticationError" in tokens[0]
    assert len(fake.calls) == 1


async def test_stream_con_rate_limit_persistente_devuelve_error_controlado():
    fake = FakeAnthropic(*[rate_limit(anthropic) for _ in range(3)])
    tokens = await collect(make_client(AnthropicClient, fake, max_retries=2).generate_stream(PREGUNTA))
    assert tokens == [f"{STREAM_ERROR_PREFIX}RateLimitError: Rate limit"]
    assert len(fake.calls) == 3
