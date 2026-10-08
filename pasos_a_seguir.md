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
- **En curso: caché de respuestas.** 8.0 (decisiones) ✅. **Siguiente: 8.1.**

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
| El tipo del modelo (`dtype`) | `--dtype` | **no entra** (decisión D1) |
| El modelo, el prompt, el presupuesto de tokens, `MAX_NEW_TOKENS`… | `src/generator.py` | en el **namespace**: el hash del texto de `generator.py` |

Dos consecuencias importantes:

1. **Meter el texto de los chunks en la clave** hace que nunca haya un acierto falso: si un fichero cambia, su texto cambia y la clave cambia. Además, `index` borra la caché cuando cambia algo (decisión D4): el texto en la clave es una segunda protección, por ejemplo si alguien edita un fichero y usa el índice viejo sin reindexar.
2. **El namespace es el hash de `generator.py` leído como texto, sin importarlo.** Si cambias el prompt o cualquier constante del generador, el namespace cambia y las respuestas viejas se ignoran solas. Y como no se importa `generator.py`, comprobar la caché **nunca carga PyTorch**.

La clave final: `sha256` de un JSON con `{pregunta, [[ruta, inicio, fin, texto], ...]}` → 64 caracteres hexadecimales.

## Diseño

- Clase **`AnswerCache`** en `src/answer_cache.py` (sin torch). Mismo patrón que `CachedRetriever`: un vacío si el fichero falta, está roto o es de otro namespace; escritura atómica (`.tmp` + `os.replace`); `flush()` solo escribe si hay algo nuevo; si no se puede escribir, aviso y sigue.
- Modelo pydantic **`AnswerCacheData`** (`namespace`, `entries: dict[str, str]`) en `models.py`.
- Fichero: `<index_dir>/answer_cache.json`, junto a `query_cache_<retriever>.json` (decisión D2).
- Interruptor: el `--cache` de siempre, `True` por defecto (decisión D3).
- Se borra en `IncrementalIndexer._save`, junto a las cachés de búsquedas (decisión D4).
- Lo usan los tres sitios que generan: `RagCLI.answer`, `RagCLI.answer_dataset` (`_answer_all`) y `RagApi.answer`.
- Sin fuentes (`NO_SOURCES`) no se cachea nada: ya no carga el modelo.

## 8.0 — Decisiones ✅ (tomadas por el usuario, 2026-10-08)

- [x] **D1 — El `dtype` no entra en la clave.** `--dtype auto`, el valor por defecto, se adapta a la máquina (float32 o bfloat16 según la memoria), y la respuesta es prácticamente la misma con los dos tipos. Meterlo solo partiría la caché sin ganar nada útil. Consecuencia: si un día se fuerza otro `--dtype`, se puede recibir una respuesta generada con el otro tipo; es aceptable. Documentarlo en el README como decisión de diseño.
- [x] **D2 — Vive donde las otras cachés:** `<index_dir>/answer_cache.json`, junto a `query_cache_bm25.json` y `query_cache_hybrid.json`. `answer` y `serve` ya tienen `--index_dir`. **`answer_dataset` no lo tiene: se le añade `--index_dir`** (por defecto `data/processed`), documentado como la carpeta de la caché de respuestas.
- [x] **D3 — El mismo interruptor `--cache`, `True` por defecto**, para las dos cachés a la vez. `--cache=False` desactiva búsquedas y respuestas. **`answer_dataset` gana `--cache`** (hoy no lo tiene). En la API, `serve --cache=False` crea `RagApi` sin caché de respuestas.
- [x] **D4 — Se borra solo cuando `index` cambia algo.** En `IncrementalIndexer._save`, junto al borrado de `query_cache_*.json` (`src/incremental_indexer.py`, el bucle sobre `RetrieverMode`): `_save` solo se ejecuta cuando hay cambios, así que sin cambios ("Index up to date") la caché se conserva, y con cualquier cambio (o `--full`) se borra entera. Usar `AnswerCache.file_path(self._index_dir)` y el mismo `PathError` si no se puede borrar.

## Pasos

### 8.1 — El modelo `AnswerCacheData` (SIGUIENTE)
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
    def key(question: str, chunks: list[Chunk]) -> str:
        # json.dumps de {question, [[file_path, first, last, text], ...]}
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
- `_read` y `flush` son casi iguales a los de `CachedRetriever`: copiar el patrón (y ver 8.8).

Prueba (sin torch):
```bash
uv run python -c "
from src.answer_cache import AnswerCache
from src.models import Chunk
c = [Chunk(file_path='data/raw/a.md', first_character_index=0, last_character_index=5, text='hello')]
c2 = [Chunk(file_path='data/raw/a.md', first_character_index=0, last_character_index=5, text='HELLO')]
k1 = AnswerCache.key('What?', c)
print(len(k1), k1 == AnswerCache.key('What?', c))
print(k1 != AnswerCache.key('Why?', c), k1 != AnswerCache.key('What?', c2))
print(len(AnswerCache.namespace()))
"
```
**Esperado:** `64 True`, `True True`, `64`.

- [ ] Tests (`tests/test_answer_cache.py`): la clave cambia con la pregunta, el orden de los chunks, su texto y su rango, y es la misma en llamadas repetidas; el namespace cambia si cambia `generator.py` (apuntar `GENERATOR_FILE` a un fichero temporal); ida y vuelta a disco; fichero roto, vacío o de otro namespace = caché vacía; `flush` sin cambios no escribe; carpeta sin permiso = aviso; **nunca se importa `src.generator`** (comprobar `sys.modules`).

### 8.3 — `answer` (CLI)
- [ ] Tras buscar, si hay `hits` y `--cache`: calcular la clave con `[hit.chunk for hit in hits]`. Acierto → respuesta de la caché, mensaje en stderr (`Answer read from the cache`), **sin importar PyTorch**. Fallo → generar como ahora, `put` y `flush`.
- [ ] Test: dos `answer` iguales con un generador falso (el truco de `sys.modules` de `test_api.py`) → la segunda no carga el generador; `--cache=False` siempre genera.

### 8.4 — `answer_dataset`
- [ ] Nuevas opciones `--index_dir` (por defecto `data/processed`, solo para la caché) y `--cache` (por defecto `True`), con sus docstrings (D2, D3).
- [ ] Primero construir los chunks de **todas** las preguntas (ya lo hace `_load_chunks`), calcular sus claves y separar aciertos de fallos.
- [ ] **Cargar PyTorch y Qwen solo si hay algún fallo.** La barra de progreso, solo sobre los fallos.
- [ ] Guardar la caché al final (y quizá cada N preguntas, para no perder el trabajo si se corta: decidir).
- [ ] Test: `--cache=False` siempre genera; todas cacheadas → PyTorch no se importa; mitad y mitad → el generador solo recibe las que faltan; el orden de la salida no cambia.

### 8.5 — La API (`RagApi`)
- [ ] `serve` crea la `AnswerCache` en `<index_dir>` y se la pasa a `RagApi`; con `--cache=False`, `None` (D3).
- [ ] **Un tercer lock** para la caché (es un diccionario compartido entre hilos, como la de búsquedas). Flujo: mirar la caché (lock de caché) → si falla, lock del generador → **volver a mirar** (otro hilo pudo generarla mientras se esperaba) → generar → guardar (lock de caché).
- [ ] Nunca coger el lock de la caché dentro del del generador al revés: siempre en el mismo orden, para no tener deadlock.
- [ ] Tests: segunda petición igual → el generador falso no se llama; dos peticiones iguales a la vez → una sola generación; `/search` no espera.

### 8.6 — `index` borra la caché de respuestas
- [ ] En `IncrementalIndexer._save`, junto al bucle que borra `query_cache_*.json`, borrar también `AnswerCache.file_path(self._index_dir)` (D4).
- [ ] Tests en `tests/test_incremental_indexer.py`: sin cambios ("Index up to date") la caché de respuestas sigue ahí; con un fichero cambiado, o con `--full`, se borra.

### 8.7 — Medir y documentar
- [ ] Medir `answer` sin caché y con caché (misma pregunta dos veces), y `answer_dataset` de 10 preguntas dos veces. Lo ejecuta el usuario.
- [ ] README: fila en *Caching (bonus 4)*; tabla de comandos (`answer_dataset --index_dir --cache`; `index` también borra `answer_cache.json`); decisiones de diseño (clave con el texto y sin `dtype`, namespace = hash de `generator.py`, se borra cuando el índice cambia); limitación (con otro `--dtype` forzado se puede reutilizar una respuesta del otro tipo); tests.

### 8.8 — (Opcional) Una base común para las dos cachés
- [ ] `CachedRetriever` y `AnswerCache` comparten leer-o-vacío, namespace y escritura atómica. Una clase base (`JsonFileCache`) evitaría el código repetido. Hacerlo **al final**, con los tests de las dos cachés pasando antes y después.

## Qué NO hacer
- No importar `src.generator` para calcular la clave o el namespace: el acierto tiene que ser instantáneo.
- No usar la caché de búsquedas como clave (guarda ids de chunks, que cambian al reindexar): la clave usa el **texto**.
- No cachear con `do_sample=True` ni con thinking mode: dejaría de ser correcto. Si algún día se cambia, el hash de `generator.py` lo invalida solo.
