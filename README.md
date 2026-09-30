# Unified Async LLM Client

Pre-Entrega 1 de AI Engineering (Coderhouse). Un cliente asíncrono que habla con **OpenAI** o **Anthropic** a través de una misma interfaz, con streaming, validación con Pydantic, reintentos ante fallas transitorias y fallback automático al otro proveedor.

```python
manager = AsyncLLMManager.from_env()          # LLM_PROVIDER y, si está, LLM_FALLBACK_PROVIDER
respuesta = await manager.generate(mensajes)   # ModelResponse validado, nunca una excepción
async for token in manager.generate_stream(mensajes):
    print(token, end="")
```

## Cómo correrlo

Requiere **Python 3.12** (se usa `asyncio.timeout`, disponible desde 3.11).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # y completá al menos una API key
python main.py
```

`main.py` pregunta "¿Qué es la entropía?" en modo normal y en streaming. Si están las dos keys, usa los dos proveedores (en modo normal, las dos llamadas salen a la vez con `asyncio.gather`). Si hay una sola, usa la que indique `LLM_PROVIDER`. Al final hace dos pruebas de resiliencia: una llamada con una API key inválida a propósito, para mostrar que el programa no se cae, y la misma llamada con Anthropic de respaldo, para mostrar que el fallback responde solo (esta última necesita `ANTHROPIC_API_KEY`).

## Variables de entorno

| Variable | Obligatoria | Default | Para qué |
|---|---|---|---|
| `LLM_PROVIDER` | no | `openai` | Proveedor que elige el Factory: `openai` o `anthropic` |
| `LLM_FALLBACK_PROVIDER` | no | (sin fallback) | Proveedor de respaldo si el principal falla. Tiene que ser distinto del principal y necesita su key |
| `OPENAI_API_KEY` | si usás OpenAI | | Key de OpenAI |
| `ANTHROPIC_API_KEY` | si usás Anthropic | | Key de Anthropic |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | no | `claude-haiku-4-5-20251001` | Modelo de Anthropic |
| `LLM_TEMPERATURE` | no | `0.7` | Entre 0 y 2 (Anthropic no la usa, ver abajo) |
| `LLM_MAX_TOKENS` | no | `1024` | Mayor que 0 |
| `LLM_TIMEOUT_S` | no | `30` | Timeout por llamada, en segundos |
| `LLM_MAX_RETRIES` | no | `3` | Reintentos ante errores transitorios (0 a 10) |
| `LOG_LEVEL` | no | `INFO` | Nivel de logging de `main.py` |

Si falta la key del proveedor elegido, el programa lo dice con el nombre exacto de la variable. Un valor fuera de rango (por ejemplo `LLM_TEMPERATURE=5`) lo frena Pydantic al arrancar, antes de gastar un solo request.

## Tests

```bash
pytest
```

Los tests **no usan internet ni API keys**: reemplazan los SDKs por objetos falsos que siguen un guion (responder, tirar un error, tardar). Los errores que tiran son los reales de `openai` y `anthropic` (`RateLimitError`, `AuthenticationError`, `APIConnectionError`…), así que la clasificación entre transitorio y permanente se prueba tal como funcionaría en producción.

| Grupo | Qué prueba |
|---|---|
| Validación | `temperature` fuera de 0–2, `max_tokens` ≤ 0, key vacía, rol inválido y mensaje vacío se rechazan. La key no aparece al imprimir la config. Los modelos por defecto son los vigentes. Una respuesta sin error y sin contenido no es válida. |
| Factory | `LLM_PROVIDER` elige la clase correcta (sin importar mayúsculas ni espacios), usa `openai` por defecto, lee el modelo y los parámetros del entorno, rechaza proveedores desconocidos y avisa qué key falta. |
| Traducción a cada SDK | OpenAI recibe `system` como mensaje y recibe `temperature`. Anthropic recibe `system` en su parámetro aparte, sin `temperature` y con `max_tokens`. |
| Resiliencia | Un 429 se reintenta y se recupera. Un 429 persistente devuelve error tras exactamente `1 + max_retries` llamadas. Una key inválida (401) no se reintenta: una sola llamada. Un error de red se reintenta. Un timeout cuenta como transitorio. El 529 "overloaded" de Anthropic se reintenta. El backoff crece exponencialmente. |
| Fallback | Si el principal agota los reintentos por rate limit, responde el respaldo. Un error permanente pasa al respaldo sin reintentar. Si el principal responde, no se llama al respaldo. Si fallan los dos, el error informa ambos. En streaming, hay fallback solo si el principal falla antes del primer token. El Factory arma el respaldo desde `LLM_FALLBACK_PROVIDER`, avisa si falta su key y rechaza que sea igual al principal. |
| Streaming | Los tokens llegan en orden con los dos proveedores. Si falla antes del primer token, reintenta. Si falla después, no reintenta y cierra con un chunk de error. Una key inválida o un rate limit persistente terminan en un chunk de error, sin excepción. |

## Estructura

```
schemas.py         Provider, Role, ChatMessage, LLMConfig, ModelResponse (Pydantic)
clients.py         BaseLLMClient (ABC) + OpenAIClient + AnthropicClient
manager.py         AsyncLLMManager: Factory según LLM_PROVIDER + fallback según LLM_FALLBACK_PROVIDER
main.py            Demo: modo normal, streaming y prueba con key inválida
tests/test_cliente.py
```

## Decisiones de diseño

**Una ABC con la resiliencia adentro.** `BaseLLMClient` define la API pública (`generate` y `generate_stream`) e implementa ahí, una sola vez, los reintentos, los timeouts y el logging. Cada proveedor solo implementa dos métodos abstractos, `_complete` y `_stream`, que traducen al idioma de su SDK. Así la política de errores no se duplica y es igual para los dos proveedores (patrón *Template Method*).

**Factory con un registro.** `AsyncLLMManager` tiene un diccionario `Provider → clase`. Cambiar de proveedor es cambiar `LLM_PROVIDER`; sumar uno nuevo es escribir su cliente y agregar una línea al registro.

**Fallback en el manager, no en los clientes.** Cada cliente se ocupa de sus propios reintentos. Si aun así falla (reintentos agotados o error permanente, como una key inválida o un modelo retirado), `AsyncLLMManager` repite el pedido con el proveedor de `LLM_FALLBACK_PROVIDER`, sin intervención humana. Así cada pieza tiene una sola responsabilidad: el cliente sabe reintentar y el manager sabe a quién recurrir. En streaming, el fallback solo ocurre si el principal falla antes del primer token, por el mismo motivo que los reintentos.

**Errores controlados, nunca un crash.** `generate()` siempre devuelve un `ModelResponse`: si algo falló, `ok` es `False` y `error` dice qué pasó. `generate_stream()` nunca lanza: si falla, el último chunk empieza con `STREAM_ERROR_PREFIX` (`"\n[ERROR] "`), así el consumidor puede detectarlo.

**Qué se reintenta y qué no.**

| Tipo | Ejemplos | Qué hace el cliente |
|---|---|---|
| Transitorio | 429, timeout, error de conexión, 5xx, 529 (Anthropic sobrecargado) | Reintenta con backoff exponencial y jitter: ~0,5 s, 1 s, 2 s… |
| Permanente | 401 key inválida, 400 pedido mal armado, 404 modelo inexistente | Devuelve el error enseguida: reintentar va a fallar igual |

**Timeouts.** En modo normal hay un `asyncio.timeout` por intento. En streaming, el timeout es por chunk: corta si el primer token (o el siguiente) no llega a tiempo, pero no corta una respuesta larga que sigue fluyendo.

**Logging estructurado** en formato `clave=valor`: proveedor, intento, tipo de error, espera del backoff, latencia, TTFT (tiempo hasta el primer token) y cada conmutación de fallback.

## Errores de la pista oficial que este proyecto evita

- **Anthropic no recibe `temperature`.** El SDK actual de Anthropic (>= 1.0) da `TypeError` si se la pasás: `messages.create()` y `messages.stream()` ya no tienen ese parámetro (se verificó con `inspect.signature` en anthropic 1.9). `LLMConfig` la sigue validando porque OpenAI sí la usa.
- **El rol `system` va aparte en Anthropic.** Anthropic no acepta `role="system"` dentro de `messages`. `AnthropicClient` junta los mensajes de sistema y los manda en el parámetro `system`.
- **Modelos vigentes y configurables.** La pista usa `claude-3-5-sonnet-20241022`, que está retirado. Acá los modelos salen del `.env`, con `gpt-4o-mini` y `claude-haiku-4-5-20251001` por defecto.
- **Reintentos y fallback.** La pista solo captura el error y lo devuelve; acá los transitorios se reintentan con backoff exponencial y, si el proveedor sigue fallando, responde el de respaldo.
- **`max_retries=0` en los SDKs.** Los SDKs de OpenAI y Anthropic reintentan 2 veces por su cuenta. Si además reintentamos nosotros, los intentos se multiplican (3 × 3 = 9) sin que se note. Por eso se crean con `max_retries=0` y la política es solo la nuestra.
- **En streaming, solo se reintenta antes del primer token.** Si el usuario ya vio medio texto, reintentar lo repetiría desde el principio. En ese caso se corta y se informa el error.
