"""Demo: la misma pregunta en modo normal y en streaming, con cada proveedor configurado."""
import asyncio
import logging
import os

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import API_KEY_ENV, ChatMessage, LLMConfig, Provider, Role

PREGUNTA = [
    ChatMessage(role=Role.SYSTEM, content="Respondé en español, en dos oraciones como máximo."),
    ChatMessage(role=Role.USER, content="¿Qué es la entropía?"),
]


def crear_managers() -> list[AsyncLLMManager]:
    """Ambos proveedores si están las dos keys; si no, el que indique LLM_PROVIDER."""
    if all(os.getenv(var) for var in API_KEY_ENV.values()):
        return [AsyncLLMManager.from_env(p) for p in Provider]
    return [AsyncLLMManager.from_env()]


async def modo_normal(managers: list[AsyncLLMManager]) -> None:
    # gather: las llamadas viajan a la vez; el total es lo que tarda la más lenta, no la suma.
    respuestas = await asyncio.gather(*(m.generate(PREGUNTA) for m in managers))
    for r in respuestas:
        texto = r.content if r.ok else f"❌ {r.error}"
        print(f"\n[{r.provider.value} · {r.model} · {r.attempts} intento(s) · {r.latency_ms:.0f} ms]")
        print(texto)


async def modo_streaming(managers: list[AsyncLLMManager]) -> None:
    # De a uno, para que los tokens de dos proveedores no se mezclen en pantalla.
    for m in managers:
        print(f"\n[{m.config.provider.value} · streaming]")
        async for token in m.generate_stream(PREGUNTA):
            print(token, end="", flush=True)
        print()


async def prueba_key_invalida() -> None:
    config = LLMConfig(provider=Provider.OPENAI, api_key="sk-invalida-a-proposito")
    r = await AsyncLLMManager(config).generate(PREGUNTA)
    print("¿El programa siguió vivo? ✅ Sí")
    print(f"Error controlado ({r.attempts} intento/s): {r.error}")


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # sin esto, httpx2 (el HTTP de los SDKs) loguea cada request

    try:
        managers = crear_managers()
    except ValueError as e:  # falta una key o LLM_PROVIDER es inválido: mensaje claro, sin traceback
        print(f"⚠️ No se pudo configurar el proveedor: {e}")
        managers = []

    if managers:
        print("=== Modo normal ===")
        await modo_normal(managers)
        print("\n=== Modo streaming ===")
        await modo_streaming(managers)

    print("\n=== Prueba de resiliencia: API key inválida ===")
    await prueba_key_invalida()


if __name__ == "__main__":
    asyncio.run(main())
