# Pasos a seguir: caché de respuestas

> **Para quien retome esto (otro agente o yo):** lee primero *Cómo trabajar con el usuario* y *Dónde estamos*. Este fichero está en `.gitignore`: son notas de trabajo, no se entregan.

## Cómo trabajar con el usuario

- Responder en **español**; README, docstrings y mensajes del programa en **inglés**.
- **Nunca hacer commits** ni `amend`: los hace el usuario. Se le puede proponer el mensaje.
- Proyecto **100 % OOP** (nada de funciones sueltas).
- **No ejecutar Qwen ni el pipeline** (`index`, `search`, `answer`…) salvo que lo pida. **No lanzar la suite entera de pytest**: el ordenador va justo de RAM; ejecutar solo los ficheros de test afectados, o `pytest --collect-only` para contar.
- Él programa los pasos y pide revisiones. **Cómo le gusta que se le guíe:**
  - un paso pequeño cada vez, con **para qué sirve** antes del cómo;
  - un **esqueleto** con comentarios de lo que va en cada método, y la pieza concreta cuando se atasca;
  - un **comando de prueba** con la salida esperada (él lo ejecuta);
  - al revisar: primero lo que rompe, luego lo que da resultados malos sin error, luego detalles; citar `fichero:línea`;
  - tras cada paso que funciona: marcarlo aquí y decir cuál es el siguiente.
- Los tests y la documentación los suele pedir hechos: seguir el estilo existente (docstring en cada test, clases por tema).
- Antes de afirmar que algo es más rápido, **medirlo** (y repetir: la máquina es compartida).

## Dónde estamos (2026-10-08)

- Parte obligatoria y los **cinco bonus** hechos. 1,297 tests, `make lint` limpio.
- `test_corpus` movido a `tests/test_corpus/` (datasets regenerados con la ruta nueva; excluido de pytest, flake8 y mypy). `data/`, el PDF, `DEVLOG.md` y este fichero están en `.gitignore`.
- **Siguiente: caché de respuestas**, paso **8.0** (decisiones) y luego **8.1**.

---

# Caché de respuestas

## Para qué sirve

Hoy, preguntar dos veces lo mismo cuesta lo mismo las dos veces:

| | Primera vez | Segunda vez |
|---|---|---|
| `answer` (CLI) | cargar PyTorch + Qwen (~6 s) + generar (~15 s) | **igual** |
| `/answer` (API) | cargar + generar | generar (~15 s) |
| `answer_dataset` (100 preguntas) | ~28 min | **igual** |

La caché de búsquedas (`CachedRetriever`) solo ahorra la búsqueda (milisegundos). Lo caro es **generar**. Con una caché de respuestas, repetir una pregunta sería instantáneo y **sin importar PyTorch**.

**Por qué es correcto cachear:** la generación es determinista (`do_sample=False`): la misma entrada da siempre la misma respuesta. Una respuesta cacheada es exactamente la que se habría generado.

## La idea clave: qué decide la respuesta

La respuesta depende de **lo que entra al modelo** y de **cómo es el modelo**:

| Qué | Dónde está | Cómo entra en la caché |
|---|---|---|
| La pregunta | `query` | en la **clave** |
| Los chunks del contexto, en orden: ruta, rango **y texto** | los `hits` / `_load_chunks` | en la **clave** |
| El tipo del modelo (`dtype`): float32 y bfloat16 pueden dar respuestas distintas | `--dtype` | en la **clave** |
| El modelo, el prompt, el presupuesto de tokens, `MAX_NEW_TOKENS`… | `src/generator.py` | en el **namespace**: el hash del texto de `generator.py` |

Dos consecuencias importantes:

1. **Meter el texto de los chunks en la clave** hace que la caché sea válida aunque cambie el índice: si un fichero cambia, su texto cambia, la clave cambia y no hay acierto falso. Las respuestas de los chunks que no cambiaron **siguen sirviendo** después de reindexar (encaja con el indexado incremental).
2. **El namespace es el hash de `generator.py` leído como texto, sin importarlo.** Si cambias el prompt o cualquier constante del generador, el namespace cambia y las respuestas viejas se ignoran solas. Y como no se importa `generator.py`, comprobar la caché **nunca carga PyTorch**.

La clave final: `sha256` de un JSON con `{pregunta, dtype, [[ruta, inicio, fin, texto], ...]}` → 64 caracteres hexadecimales.

## Diseño

- Clase **`AnswerCache`** en `src/answer_cache.py` (sin torch). Mismo patrón que `CachedRetriever`: un vacío si el fichero falta, está roto o es de otro namespace; escritura atómica (`.tmp` + `os.replace`); `flush()` solo escribe si hay algo nuevo; si no se puede escribir, aviso y sigue.
- Modelo pydantic **`AnswerCacheData`** (`namespace`, `entries: dict[str, str]`) en `models.py`.
- Fichero: `<index_dir>/answer_cache.json` (ver decisión D2).
- Lo usan los tres sitios que generan: `RagCLI.answer`, `RagCLI.answer_dataset` (`_answer_all`) y `RagApi.answer`.
- Sin fuentes (`NO_SOURCES`) no se cachea nada: ya no carga el modelo.

## 8.0 — Decisiones (las toma el usuario)

- [ ] **D1 — `dtype` en la clave.** Recomendado: el `dtype` **pedido** (`auto`, `float32`, `bfloat16`). Con `auto`, la misma máquina elige siempre el mismo tipo en la práctica; otra máquina con menos memoria podría elegir otro y reutilizar una respuesta generada con el otro tipo. Alternativa: no meter el `dtype` (más aciertos, riesgo pequeño de respuesta ligeramente distinta).
- [ ] **D2 — Dónde vive el fichero.** Recomendado: `<index_dir>/answer_cache.json` para los tres, como la caché de búsquedas. `answer` y `serve` ya tienen `--index_dir`; `answer_dataset` **no**: habría que añadirle `--index_dir` (por defecto `data/processed`), documentado como "dónde está la caché de respuestas". Alternativa: una opción `--answer_cache <ruta>` en los tres.
- [ ] **D3 — Interruptor.** Recomendado: reutilizar el `--cache` que ya existe (un interruptor para las dos cachés). `answer_dataset` lo ganaría.
- [ ] **D4 — ¿`index` borra la caché de respuestas?** Recomendado: **no**. Como la clave lleva el texto, nunca da una respuesta de un chunk que cambió, y las demás siguen siendo válidas. (La caché de búsquedas sí se borra, porque guarda ids que cambian.)

## Pasos

### 8.1 — El modelo `AnswerCacheData` (SIGUIENTE tras 8.0)
- [ ] En `models.py`, junto a `QueryCache`: `namespace: str`, `entries: dict[str, str]` (clave → respuesta), con docstring y `Attributes:`.
- [ ] Prueba: JSON de ida y vuelta; un JSON roto da `ValidationError`.

### 8.2 — La clase `AnswerCache`
Esqueleto (`src/answer_cache.py`):

```python
class AnswerCache:
    """<qué es y por qué es correcto cachear: generación determinista>"""

    FILE_NAME = "answer_cache.json"
    GENERATOR_FILE = Path(__file__).parent / "generator.py"

    @classmethod
    def file_path(cls, index_dir: Path) -> Path:
        # index_dir / FILE_NAME
        ...

    @classmethod
    def namespace(cls) -> str:
        # sha256 del TEXTO de generator.py (read_text, sin importarlo)
        ...

    @staticmethod
    def key(question: str, chunks: list[Chunk], dtype: DtypeMode) -> str:
        # json.dumps de {question, dtype, [[file_path, first, last, text], ...]}
        # con sort_keys=True, y sha256 de eso (hexdigest)
        ...

    def __init__(self, path: Path) -> None:
        # leer el fichero (vacío si falta / roto / otro namespace)
        ...

    def get(self, key: str) -> str | None: ...
    def put(self, key: str, answer: str) -> None: ...   # marca "hay cambios"
    def flush(self) -> None: ...                          # como CachedRetriever.flush
```

Pistas:
- `hashlib.sha256(texto.encode("utf-8")).hexdigest()`.
- `json.dumps(..., sort_keys=True, ensure_ascii=False)`: el mismo contenido da siempre el mismo texto, así que la misma clave.
- `_read` y `flush` son casi iguales a los de `CachedRetriever`: copiar el patrón (y ver 8.7).

Prueba (sin torch):
```bash
uv run python -c "
from src.answer_cache import AnswerCache
from src.models import Chunk
c = [Chunk(file_path='data/raw/a.md', first_character_index=0, last_character_index=5, text='hello')]
k1 = AnswerCache.key('What?', c, 'auto')
print(len(k1), k1 == AnswerCache.key('What?', c, 'auto'))
print(k1 != AnswerCache.key('What?', c, 'float32'), k1 != AnswerCache.key('Why?', c, 'auto'))
print(len(AnswerCache.namespace()))
"
```
**Esperado:** `64 True`, `True True`, `64`.

- [ ] Tests (`tests/test_answer_cache.py`): la clave cambia con la pregunta, el orden de los chunks, su texto, su rango y el `dtype`, y no cambia con lo demás; el namespace cambia si cambia `generator.py` (apuntar `GENERATOR_FILE` a un fichero temporal); ida y vuelta a disco; fichero roto, vacío o de otro namespace = caché vacía; `flush` sin cambios no escribe; carpeta sin permiso = aviso; **nunca se importa `src.generator`** (comprobar `sys.modules`).

### 8.3 — `answer` (CLI)
- [ ] Tras buscar, si hay `hits` y `--cache`: calcular la clave con `[hit.chunk for hit in hits]`. Acierto → respuesta de la caché, mensaje en stderr (`Answer read from the cache`), **sin importar PyTorch**. Fallo → generar como ahora, `put` y `flush`.
- [ ] Test: dos `answer` iguales con un generador falso (el truco de `sys.modules` de `test_api.py`) → la segunda no carga el generador; `--cache=False` siempre genera.

### 8.4 — `answer_dataset`
- [ ] Primero construir los chunks de **todas** las preguntas (ya lo hace `_load_chunks`), calcular sus claves y separar aciertos de fallos.
- [ ] **Cargar PyTorch y Qwen solo si hay algún fallo.** La barra de progreso, solo sobre los fallos.
- [ ] Guardar la caché al final (y quizá cada N preguntas, para no perder el trabajo si se corta: decidir).
- [ ] Test: todas cacheadas → PyTorch no se importa; mitad y mitad → el generador solo recibe las que faltan; el orden de la salida no cambia.

### 8.5 — La API (`RagApi`)
- [ ] `RagApi` recibe una `AnswerCache` (o `None` con `--cache=False`).
- [ ] **Un tercer lock** para la caché (es un diccionario compartido entre hilos, como la de búsquedas). Flujo: mirar la caché (lock de caché) → si falla, lock del generador → **volver a mirar** (otro hilo pudo generarla mientras se esperaba) → generar → guardar (lock de caché).
- [ ] Nunca coger el lock de la caché dentro del del generador al revés: siempre en el mismo orden, para no tener deadlock.
- [ ] Tests: segunda petición igual → el generador falso no se llama; dos peticiones iguales a la vez → una sola generación; `/search` no espera.

### 8.6 — Medir y documentar
- [ ] Medir `answer` sin caché y con caché (misma pregunta dos veces), y `answer_dataset` de 10 preguntas dos veces. Lo ejecuta el usuario.
- [ ] README: fila en *Caching (bonus 4)*, decisión de diseño (clave con el texto, namespace = hash de `generator.py`), limitación (la caché crece sin límite, como la de búsquedas), tests.

### 8.7 — (Opcional) Una base común para las dos cachés
- [ ] `CachedRetriever` y `AnswerCache` comparten leer-o-vacío, namespace y escritura atómica. Una clase base (`JsonFileCache`) evitaría el código repetido. Hacerlo **al final**, con los tests de las dos cachés pasando antes y después.

## Qué NO hacer
- No importar `src.generator` para calcular la clave o el namespace: el acierto tiene que ser instantáneo.
- No usar la caché de búsquedas como clave (guarda ids de chunks, que cambian al reindexar): la clave usa el **texto**.
- No cachear con `do_sample=True` ni con thinking mode: dejaría de ser correcto. Si algún día se cambia, el hash de `generator.py` lo invalida solo.
