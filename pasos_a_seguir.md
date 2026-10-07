# Pasos a seguir: bonus del proyecto RAG

> **Para quien retome esto (otro agente o yo):** lee primero **"Dónde estamos ahora"** justo debajo y luego **"Contexto para quien retome esto"**. Bonus 1, 2, 4 y 5: hechos (se dejan como historia de cómo se hizo). **Siguiente: bonus 3 (indexado incremental)**, luego el cierre.

## Dónde estamos ahora (2026-10-07, campus)

| Parte | Estado |
|---|---|
| Parte obligatoria | ✅ completa (recall@5 con `bm25`: docs 0.870, code 0.758); revisada contra el subject (38 entradas degeneradas sin traceback) |
| Bonus 1 (embeddings) + 2 (híbrido) | ✅ por defecto `--retriever hybrid`: docs **0.910**, code **0.828** |
| Bonus 4 (caché) | ✅ arranque en frío (`StaticEncoder`) + caché de búsquedas en disco (`--cache`, un fichero por retriever) |
| Bonus 5 (API HTTP) | ✅ `serve` + `/health`, `/search`, `/answer`; errores → 422 / 503; 36 tests; README y DEVLOG (Steps 24–25). **Sin commit** |
| Bonus 3 (indexado incremental) | **siguiente** |
| Cierre (`.gitignore`, sacar de git PDF/DEVLOG/pasos/.vscode/`dataset_mini_3.json`, repaso, preparar la demo de la defensa) | pendiente |

Estimación: **~90 %** del proyecto.

**Último commit:** `6f402fe` (API con `/health` y `/search`). **Sin commit** (el usuario lo hace; mensaje propuesto: `Add /answer, error handling, locks and tests to the HTTP API (bonus 5)`): `src/api.py`, `serve --dtype` y su docstring en `commands.py`, `tests/test_api.py`, `make serve` y cachés de mypy separadas en el `Makefile`, README, DEVLOG.

**Cómo quedó el bonus 5 (2026-10-07):**
- `RagApi(retriever, dtype)` en `src/api.py`. `serve --host --port --index_dir --retriever --dtype --cache` (o `make serve SERVE_ARGS=...`).
- `GET /answer` (no POST, por coherencia con `/search`): `retriever.search` directamente (necesita el texto de los chunks). Sin fuentes → `NO_SOURCES` sin cargar Qwen. Qwen se carga en el primer `/answer` y se guarda en `self._generator`.
- `dtype` es del servidor (`serve --dtype`), no de cada petición: el modelo se carga una vez y un `dtype` por petición se ignoraría en silencio.
- **Dos locks (mutex):** `_retriever_lock` (search + flush: `CachedRetriever` no es thread-safe) y `_generator_lock` (carga + generación). Nunca se cogen a la vez → sin deadlock, y `/search` no espera a una generación.
- **Errores:** `self.app.exception_handler(RagError)(self._rag_error)` → todo `RagError` es **503** `{"detail": "<mensaje de la CLI>"}`. Sin índice → 503, y si luego se ejecuta `index` funciona sin reiniciar (la caché reintenta la carga). PyTorch que no carga → 503 (`HTTPException`). Parámetros malos → 422; `query` de solo espacios también (`StringConstraints(strip_whitespace=True, min_length=1)`).
- Puerto ocupado → exit 3 de uvicorn: **se deja así** (decidido por el usuario, no hay traceback).
- Lentitud de `make lint` resuelta: `lint` y `lint-strict` compartían `.mypy_cache` con flags distintos y se invalidaban (~50 s cada vez). Ahora `.mypy_cache/lint` y `.mypy_cache/strict` (~3 s en caliente). No había procesos zombie; la máquina va justa de RAM (Pylance ~1.9 GB, 2 GB de swap).
- Tests: 1187 en total, `make lint` y `make lint-strict` limpios.

**Si se continúa en casa:** `data/processed` no está en git. Antes de medir nada hay que ejecutar `uv run -m src index` (crea `index.json`, `vectors.npy`, `owners.npy`, `embeddings.npy`, `tokenizer.json`; ~13 s en el campus, menos en casa). En el campus el entorno está en `/sgoinfre` (`UV_PROJECT_ENVIRONMENT=/sgoinfre/students/saperez-/rag-venv`, `HF_HOME=/sgoinfre/students/saperez-/hf`); en casa serán otros, no hace falta tocarlos.

Configuración de la búsqueda híbrida (por si hay que tocarla):

| Qué | Valor |
|---|---|
| Modelo | `minishlab/potion-base-8M` (Model2Vec, 30 MB), ejecutado con numpy por `StaticEncoder` y guardado dentro de `data/processed` |
| Ventanas | ~400 caracteres, por líneas enteras (`>` 400 cierra la ventana) |
| Texto de cada ventana | `"<palabras de la ruta>\n<ventana>"` |
| Puntuación de un chunk | la de su **mejor** ventana (máximo) |
| Pregunta | se vectoriza **tal cual**, sin quitar stopwords |
| Híbrido | reordenar los **30** primeros de BM25 con **α = 0.5** |

Consejo: un commit al terminar cada paso que funcione.

---

## Contexto para quien retome esto (otro agente o yo mismo)

### Cómo trabajar con el usuario

- Responder en **español**; README, DEVLOG, docstrings y mensajes del programa en **inglés**.
- **Nunca hacer commits** ni `amend`: el usuario los hace. Se le puede proponer el mensaje.
- Proyecto **100 % OOP** (nada de funciones sueltas). Placeholders: `pass` = lo implementa una subclase; código pendiente = `raise NotImplementedError("<name> is not implemented yet")`.
- Si hay erratas en inglés en su código o mensajes, **corregirlas antes** de citarlas en la documentación.
- Cuando pide que se programe algo es por esfuerzo/beneficio: hacerlo sin sermones, siguiendo la estructura que ya tiene el proyecto. Ahora está programando él estos pasos y pide revisiones: revisar, explicar, y solo tocar el código cuando lo pida.
- **Cómo le gusta que se le guíe** (funcionó bien en los bonus 1, 2 y 4):
  - un paso pequeño cada vez, con **para qué sirve** antes del cómo;
  - un **esqueleto** de la clase con comentarios de lo que va en cada método, y las piezas concretas (la función de numpy, la línea exacta) cuando se atasca;
  - un **comando de prueba** con la salida esperada al final de cada paso (él lo ejecuta);
  - al revisar: errores que lo rompen primero, luego lo que da resultados malos sin error, luego detalles; citar `fichero:línea`;
  - tras cada paso que funciona: marcarlo aquí con lo medido, y decir cuál es el siguiente.
- Los tests (`tests/`) y la documentación (README, DEVLOG) los suele pedir hechos: escribirlos siguiendo el estilo existente (docstring en cada test, clases por tema).
- Antes de afirmar que algo es más rápido o igual, **medirlo** (y repetir: la máquina del campus es compartida y los tiempos saltan).
- **No ejecutar Qwen** salvo que lo pida (tarda ~20 s por pregunta). El modelo de embeddings sí se puede ejecutar: es rápido.
- Medir cada cambio por separado y documentarlo (README → Performance, DEVLOG).

### Estado del proyecto (2026-10-06)

- **Código** (`src/`): `commands.py` (CLI Fire; `_load_retriever(index_dir, retriever) -> Retriever` elige según `--retriever`), `base_retriever.py` (`Retriever(ABC)` con `search(query, k)` abstracto), `bm25_index.py` (`BM25Index(Retriever)`, `chunk_count`, cada `SearchHit` lleva `chunk_id`), `vector_index.py` (`VectorIndex`: ventanas, `chunk_score`, guarda `vectors.npy` + `owners.npy` + el modelo), `static_encoder.py` (`StaticEncoder`: tokenizer + tabla con numpy, `embeddings.npy` + `tokenizer.json`), `hybrid_retriever.py` (`HybridRetriever(Retriever)`), `exceptions.py` (`VectorIndexError`, etc.).
- **Tests:** 1116, ~60 s. `tests/conftest.py` cambia el modelo de embeddings por `FakeEncoder` en **todos** los tests (nunca se carga el modelo real ni Qwen). `make lint-strict` limpio.
- **Medidas actuales** (campus, `make test_limits`): `index` 13.2 s, 199 preguntas 5.4 s; recall@5 docs 0.910 / code 0.828 (`hybrid`), 0.870 / 0.758 (`bm25`).
- **Documentación al día hasta la fase 1 del bonus 4:** README (arquitectura, tiempos, tablas de mejoras, tests, decisiones) y DEVLOG Step 22.
- Dependencias: `sentence-transformers` **quitado**; `tokenizers`, `safetensors`, `huggingface-hub` añadidos; `torch` / `transformers` siguen (Qwen).
- Modelos en la caché de Hugging Face: `minishlab/potion-base-8M` y `Qwen/Qwen3-0.6B`.
- Orden recomendado de lo que queda: bonus 4 (fases 2–4) → bonus 5 (API HTTP: recibe un `Retriever`, `CachedRetriever` le sirve tal cual) → bonus 3 (indexado incremental, el más caro).

### Qué dice el subject (y por qué el índice de vectores va dentro de `index`)

- Bonus 1: *"add a vector index built with a lightweight CPU model (such as all-MiniLM-L6-v2) next to your lexical index"*. MiniLM es un **ejemplo**, no obligatorio. Bonus 2: *"combine the lexical and semantic rankings into a single result list"*.
- *"Indexing time: at most 5 minutes for the whole corpus"* está en la parte obligatoria; el capítulo de bonus no repite límites, pero dice que todo corre en *"a CPU-only campus machine"* (≈3× más lento que el PC de casa) y que un bonus cuenta si está *"implemented and working"*.
- No hay hoja de corrección pública. Otros repos de estudiantes (OleksiiShtohrin, vinsongqj, Sabofxx) construyen los vectores con MiniLM **aparte** del `index` obligatorio por el coste en CPU.
- Con `potion-base-8M` el índice de vectores tarda segundos, así que **cabe dentro de `index`** sin discusión.

### Cómo se llegó a la configuración elegida

Todas las cifras: recall@k con `src.evaluator.Evaluator` (las mismas que la moulinette) sobre los datasets públicos `data/datasets/{Unanswered,Answered}Questions/dataset_{docs,code}_public.json`, k = 10, chunks de 2000 caracteres, el índice BM25 de `data/processed`. Tablas completas en el README: *Semantic embeddings: choosing the model* y *Semantic embeddings: tuning `potion-base-8M`*.

1. **Velocidad de MiniLM** (`all-MiniLM-L6-v2`, 256 tokens): ~20 chunks/s → ~400 s el corpus (12 907 chunks) en casa, ~20 min en el campus. Con 128 tokens: 344 s. **No cabe en 5 min.**
2. **Calidad con chunks enteros** (vectores solos, recall@5 docs / code): MiniLM-256 0.51 / 0.374, MiniLM-128 0.46 / 0.333, `potion-base-8M` 0.44 / 0.343, `static-retrieval-mrl-en-v1` 0.35 / 0.293, `potion-retrieval-32M` 0.34 / 0.263. Mezclados con BM25 por RRF (`peso / (60 + posición)`, pesos 1 / 0.5 / 0.3, y solo documentación): **todos empeoran a BM25**.
3. **Modelo elegido: `potion-base-8M`**: casi igual que MiniLM-128 y el corpus en ~2 s. FastText descartado sin probar (medias de palabras, no entrenado para búsqueda, ~7 GB).
4. **Ajuste de `potion-base-8M`** (lo que lo hizo útil): el problema era la **media** de los vectores en chunks largos.
   - Ventanas de ~400 caracteres + máximo por chunk: docs 0.44 → 0.70.
   - Ruta del fichero (palabras) delante de cada ventana: code 0.343 → 0.404.
   - Las dos juntas: **0.73 / 0.596** (59 778 ventanas, ~3 s).
   - Quitar stopwords de la pregunta **empeora** 1–2 puntos (al revés que en BM25).
5. **Combinación con BM25**: RRF sigue sin ganar (como mucho igual que BM25). **Reordenar los N primeros de BM25** con `(1 − α) × BM25 / mejor BM25 + α × similitud` sí gana: los vectores solo reordenan candidatos que BM25 ya encontró, no pueden colar chunks irrelevantes.
6. **Barrido** N ∈ {10, 20, 30, 50} × α ∈ {0.3 … 0.8} × pregunta con/sin stopwords (48 combinaciones): toda la zona N 20–50, α 0.3–0.7 supera a BM25 (docs 0.89–0.92, code 0.78–0.83). Elegido el **centro** (N = 30, α = 0.5), no la mejor celda: **docs 0.910, code 0.828** (@10: 0.920 / 0.859).
7. Salvedad: ajustado y medido sobre los mismos datasets públicos; la zona estable hace poco probable el sobreajuste, pero los privados pueden dar algo menos.

### Código de referencia del experimento

Para reproducir las cifras o comparar con la implementación (los scripts originales estaban en una carpeta temporal). Ejecutar desde la raíz con `PYTHONPATH=. uv run python <script>.py`; tarda ~1 min.

```python
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.bm25_index import BM25Index
from src.tokenizer_bm import Tokenizer
from src.models import (RagDataset, StudentSearchResults,
                        MinimalSearchResults, MinimalSource)
from src.evaluator import Evaluator

idx = BM25Index.load(Path("data/processed"))
chunks = idx._chunks
model = SentenceTransformer("minishlab/potion-base-8M", device="cpu")
tok = Tokenizer()

def windows(text, size=400):          # líneas enteras; '>' size cierra
    out, cur = [], ""
    for line in text.splitlines(keepends=True):
        if cur and len(cur) + len(line) > size:
            out.append(cur); cur = ""
        cur += line
    if cur.strip():
        out.append(cur)
    return out or [text]

texts, owners = [], []
for i, c in enumerate(chunks):
    prefix = " ".join(tok.tokenize(c.file_path)) + "\n"
    for w in windows(c.text):
        texts.append(prefix + w); owners.append(i)
V = model.encode(texts, batch_size=512, normalize_embeddings=True)
owners = np.array(owners)

def chunk_scores(query):              # pregunta tal cual
    s = V @ model.encode([query], normalize_embeddings=True)[0]
    best = np.full(len(chunks), -1.0)
    np.maximum.at(best, owners, s)
    return best

pos = {id(c): i for i, c in enumerate(chunks)}
def hybrid(query, k, top=30, alpha=0.5):
    hits = idx.search(query, top)
    ids = [pos[id(h.chunk)] for h in hits]
    b = np.array([h.score for h in hits]); b = b / b.max()
    comb = (1 - alpha) * b + alpha * chunk_scores(query)[ids]
    return [ids[j] for j in np.argsort(-comb)[:k]]

for d in ("docs", "code"):
    f = f"dataset_{d}_public.json"
    q = RagDataset.model_validate_json(Path("data/datasets/UnansweredQuestions", f).read_text())
    a = RagDataset.model_validate_json(Path("data/datasets/AnsweredQuestions", f).read_text())
    res = [MinimalSearchResults(question_id=x.question_id, question=x.question,
               retrieved_sources=[MinimalSource(file_path=chunks[i].file_path,
                   first_character_index=chunks[i].first_character_index,
                   last_character_index=chunks[i].last_character_index)
                   for i in hybrid(x.question, 10)])
           for x in q.rag_questions]
    print(d, Evaluator().evaluate(StudentSearchResults(k=10, search_results=res), a, 10).recall)
# Esperado: docs {..., 5: 0.91, 10: 0.92}, code {..., 5: ~0.828, 10: ~0.859}
```

Diferencias conocidas con la implementación del usuario: su `_get_window` no descarta ventanas solo de espacios, por eso salen **59 828** ventanas en vez de 59 778. El efecto en el recall debería ser despreciable; si el paso 8 no da ~0.91 / ~0.83, comparar contra este script.

---

## Paso 0 — Preparar ✅

- [x] `uv add numpy` (lo importas directamente).
- [x] Commit del estado anterior.
- [x] Crear `src/vector_index.py`.

## Paso 1 — Partir un chunk en ventanas ✅

- [x] `_get_window(text) -> list[str]`: recorre `text.splitlines(keepends=True)` y acumula líneas; cierra la ventana cuando la siguiente línea haría pasar de `MAX_WINDOW_SIZE` **y** la ventana no está vacía.
- [x] Comprobado: 25 líneas de 80 caracteres → 5 ventanas de 400; primera línea de 500 → `[500, 11]`; texto vacío → `[]`; al juntar las ventanas sale el texto original.

## Paso 2 — El texto que se vectoriza ✅

- [x] `_window_chunk(chunks) -> list[WindowChunk]`: un `WindowChunk` por chunk, **en el mismo orden** (la posición en la lista es el número de chunk).
- [x] A cada ventana se le antepone `" ".join(Tokenizer().tokenize(chunk.file_path)) + "\n"`.

## Paso 3 — Construir el índice (`build`) ✅

- [x] `build` como `@classmethod`: `cls._window_chunk(chunks)`, `counts`, `owners` con `np.repeat`, lista plana `texts`, `model.encode(texts, batch_size=512, normalize_embeddings=True, show_progress_bar=True)`, `return cls(vectors, owners)`.
- [x] Import de `sentence_transformers` **dentro** de `build` (carga perezosa).
- [x] Comprobado con el corpus real: `vectors (59828, 256) float32`, `owners (59828,)` del 0 al 12906, vectores normalizados, ~8 s.
- [x] **Arreglar la línea 36**: se pegó texto dentro de `tokenizer.tokenize(chunk.file_path)`.
- [x] Usar `cls` en vez de `self` en los métodos `@classmethod`.
- [x] Tipos de `__init__`: `vectors: np.ndarray`, `owners: np.ndarray`; sin `from typing import Any`.
- [x] Constante de clase `MODEL_NAME = "minishlab/potion-base-8M"` (la usarán `build` y `search`: **el mismo** modelo para chunks y preguntas).
- [x] Atributos privados: `self._vectors`, `self._owners` (como `BM25Index`).
- [x] Orden de imports: librería estándar, luego terceros (`numpy`), luego los del proyecto.
- [x] (Opcional) `counts` con `item.windows_count`.

## Paso 4 — Guardar y cargar ✅

Métodos que quedan en `VectorIndex`: `chunk_score` (paso 5). La reordenación híbrida va en **otra clase** (paso 6).

- [x] `save(files_dir)`: `np.save` de `vectors.npy` y `owners.npy` en la misma carpeta que `index.json` (crea la carpeta si falta). `OSError` → `PathError`, como `BM25Index.save`.
- [x] `load(files_dir)` (`@classmethod`): `np.load` de los dos. Con `.npy` no hay `model_validate_json` (numpy ya devuelve el array con su forma y tipo), así que las formas se comprueban a mano:
  - falta algún fichero → `IndexNotFoundError` ("run `index` first");
  - no se puede leer → `PathError`;
  - fichero corrupto / no es `.npy` (`ValueError`) → `VectorIndexError`;
  - formas incoherentes (`owners` no es 1D, `vectors` no es 2D, `len` distintos; `ndim` antes que `len`) → `VectorIndexError`.
- [x] Error propio `VectorIndexError(RagError)` en `exceptions.py` (`DatasetError` no sirve: necesita un `ValidationError` de pydantic).
- [x] Constantes con los nombres de fichero: `VECTOR_NAME = "vectors.npy"`, `OWNERS_NAME = "owners.npy"`.
- [x] **Comprobado** (2026-10-06, arrays inventados, sin el modelo): guardar y cargar → `np.array_equal` `True` en los dos (`float32` se conserva); carpeta sin índice → `IndexNotFoundError`; `owners` de otra longitud → `VectorIndexError`; `vectors.npy` con basura → `VectorIndexError`.
- [x] Docstrings de `load`, `save` y `VectorIndexError`.

## Paso 5 — Puntuar los chunks con una pregunta ✅

- [x] Dónde vive el modelo: `_load_model()` (de la caché local primero, sin red) y `_encoder()` (lo carga la primera vez y lo guarda en `self._model`). `build` deja el modelo ya cargado en el índice.
- [x] `chunk_score(query) -> np.ndarray` (el usuario lo llamó en singular; usar este nombre en el paso 6):
  1. `query_vector = self._encoder().encode([query], normalize_embeddings=True)[0]` (pregunta **tal cual**; el `[0]` pasa de `(1, 256)` a `(256,)`).
  2. `scores = self._vectors @ query_vector` (una puntuación por ventana).
  3. `best = np.full(self._owners.max() + 1, -1.0)` y `np.maximum.at(best, self._owners, scores)` (una por chunk).
- [x] **Comprobado** (2026-10-06):
  - con un modelo falso y el ejemplo de abajo → `[0.71, 0.35, 0.48]`, forma `(3,)`;
  - con el modelo real y 3 chunks inventados → una puntuación por chunk, todas entre −1 y 1; "How do I install vLLM?" elige `docs/installation.md` y "default temperature for sampling" elige `vllm/sampling_params.py`.
- [x] Docstring de `chunk_score`.

### Cómo se asignan las puntuaciones (paso 5), con un ejemplo

Supón 3 chunks: el 0 tiene 2 ventanas, el 1 tiene 1 y el 2 tiene 2. Entonces:

```
owners = [0, 0, 1, 2, 2]          # ventana i → chunk owners[i]
vectors.shape = (5, 256)          # una fila por ventana, longitud 1 cada una
```

**1. La pregunta, a vector.** `query_vector = encoder.encode([query], normalize_embeddings=True)[0]` → forma `(256,)`, longitud 1.

**2. Una puntuación por ventana.** Como todos los vectores tienen longitud 1, el producto escalar **es** la similitud del coseno (1 = mismo sentido, 0 = nada que ver):

```
scores = self._vectors @ query_vector       # forma (5,)
scores = [0.20, 0.71, 0.35, 0.10, 0.48]     # ventana 0, 1, 2, 3, 4
```

`@` es la multiplicación de matrices: hace las 5 multiplicaciones de golpe, sin bucle.

**3. De ventanas a chunks: el máximo de cada chunk.**

```
best = np.full(3, -1.0)                      # [-1.0, -1.0, -1.0] (el coseno nunca baja de -1)
np.maximum.at(best, self._owners, scores)
```

`np.maximum.at` recorre las ventanas y, para cada una, hace `best[owners[i]] = max(best[owners[i]], scores[i])`:

| Ventana i | owners[i] | scores[i] | best después |
|---|---|---|---|
| 0 | 0 | 0.20 | [0.20, −1, −1] |
| 1 | 0 | 0.71 | [**0.71**, −1, −1] |
| 2 | 1 | 0.35 | [0.71, **0.35**, −1] |
| 3 | 2 | 0.10 | [0.71, 0.35, **0.10**] |
| 4 | 2 | 0.48 | [0.71, 0.35, **0.48**] |

Resultado: `best = [0.71, 0.35, 0.48]`, la puntuación de cada chunk es la de su **mejor** ventana. No vale `best[owners] = np.maximum(...)` normal: con índices repetidos solo se quedaría la última asignación; `.at` las aplica todas.

`n_chunks` es `self._owners.max() + 1` (aquí 2 + 1 = 3).

## Paso 6 — Reordenar los resultados de BM25 (bonus 2) ✅

- [x] Clase aparte `HybridRetriever` en `src/hybrid_retriever.py`, **no** dentro de `VectorIndex`: necesita los dos índices, y así `VectorIndex` no depende de `BM25Index`. BM25 busca por palabras, `VectorIndex` puntúa por significado y `HybridRetriever` combina.
- [x] `__init__(bm25, vector)`: guarda los dos índices en `self._bm25` y `self._vector`.
- [x] Propiedad pública `chunk_count` en `BM25Index` y `VectorIndex` (los atributos `_owners` / `_n_chunks` son privados).
- [x] Comprueba que coinciden: `vector.chunk_count != bm25.chunk_count` → `VectorIndexError` ("Run 'index' again").
- [x] Constantes de clase: `CANDIDATES = 30`, `ALPHA = 0.5`.
- [x] `search(query, k) -> list[SearchHit]`, la **misma** firma que `BM25Index.search`:
  1. `hits = self._bm25.search(query, CANDIDATES)`; vacío → `[]` (sin calcular `chunk_score`).
  2. Para cada hit: `(1 − ALPHA) × hit.score / hits[0].score + ALPHA × best[hit.chunk_id]`.
  3. `hit.model_copy(update={"score": combined})` (no modifica el hit de BM25), ordenar de mayor a menor (`sort` es estable: los empates mantienen el orden de BM25) y devolver los `k` primeros.
- [x] El id de cada chunk: `SearchHit.chunk_id` (opción A), rellenado en `BM25Index.search`.
- [x] **Comprobado** (2026-10-06, script en el scratchpad):
  - ejemplo de abajo (BM25 falso) → orden 2, 7, 9 con 0.760 / 0.650 / 0.600; `k = 2` → 2, 7;
  - `ALPHA = 0` → **exactamente** el orden de BM25 (BM25 real, 3 preguntas); `ALPHA = 1` → orden solo por la similitud;
  - sin resultados de BM25 → `[]`; índices de distinto tamaño → `VectorIndexError`; el hit original de BM25 no cambia;
  - BM25 + `VectorIndex` reales con 10 chunks inventados → resultados con sentido;
  - `make lint` (flake8 + mypy) limpio.
- [ ] (Paso 9) Docstrings de `HybridRetriever` y anotar `scored_hits: list[SearchHit]`.

### Cómo se combinan las puntuaciones (paso 6), con un ejemplo

Las dos puntuaciones están en **escalas distintas**: BM25 no tiene máximo (puede dar 3.2, 15.7, 41.0…) y la similitud va de −1 a 1. Si las sumas tal cual, BM25 lo decide todo. Por eso primero se lleva BM25 a 0–1 dividiendo por la mejor puntuación de la lista.

BM25 devuelve 3 candidatos (en la práctica, 30), con `best` del paso 5:

| Candidato (chunk) | BM25 `hit.score` | BM25 / mejor | `best[chunk]` | 0.5 × BM25 norm. + 0.5 × best |
|---|---|---|---|---|
| 7 | 12.0 | 12 / 12 = **1.00** | 0.30 | 0.5 + 0.15 = **0.650** |
| 2 | 10.8 | 10.8 / 12 = **0.90** | 0.62 | 0.45 + 0.31 = **0.760** |
| 9 | 6.0 | 6 / 12 = **0.50** | 0.70 | 0.25 + 0.35 = **0.600** |

Orden nuevo: **2, 7, 9** (BM25 decía 7, 2, 9). El chunk 2 sube porque su texto se parece más a la pregunta, aunque BM25 lo puntuara un poco menos.

- `ALPHA = 0` → solo cuenta BM25 → mismo orden que BM25 (el test del paso 6).
- `ALPHA = 1` → solo cuentan los vectores (entre los 30 candidatos).
- `ALPHA = 0.5` → mitad y mitad (lo medido: docs 0.910, code 0.828).

En código:

```
hits = bm25.search(query, CANDIDATES)          # list[SearchHit], ya ordenados
best = vectors.chunk_score(query)             # una puntuación por chunk
top_bm25 = hits[0].score                       # la mayor (la primera)
combined = [(1 - ALPHA) * hit.score / top_bm25 + ALPHA * best[id del chunk] for hit in hits]
```

y ordenar los `hits` por `combined`, de mayor a menor, quedándose con los `k` primeros. El "id del chunk" es su posición en la lista de chunks de BM25 (paso 6, punto anterior).

**Hecho (opción A): `SearchHit.chunk_id`**, rellenado en `BM25Index.search`. Antes `SearchHit` no llevaba el id del chunk (solo `chunk` y `score`), y `BM25Index.search` lo tiene dentro (`chunk_id`) pero no lo devuelve. Opciones:
- añadir un campo `chunk_id: int` a `SearchHit` y rellenarlo en `BM25Index.search` (lo más limpio: un dato que ya existe, sin buscarlo después);
- o, en `HybridRetriever`, un diccionario `{id(chunk): posición}` hecho una vez en `__init__` (funciona, pero depende de que sean los mismos objetos).

## Paso 7 — Conectarlo a la CLI ✅

- [x] `index`: después de BM25, `VectorIndex.build(chunks)` + `save` en la misma `--index_dir`, con **los mismos chunks en el mismo orden**. `index` **no** tiene `--retriever`: siempre construye los dos.
- [x] `search`, `search_dataset`, `answer`: opción `--retriever` con `RetrieverMode = Literal["bm25", "hybrid"]` en `models.py` (como `DtypeMode`). `answer_dataset` no busca (lee los resultados de `search_dataset`): sin cambios.
- [x] Un solo sitio que elige: `RagCLI._load_retriever(index_dir, retriever) -> BM25Index | HybridRetriever`. `_search_question` recibe `indexer: BM25Index | HybridRetriever`. (La clase base `Retriever` y la capa de servicio, más adelante, antes de los bonus 4 y 5.)
- [x] Por defecto: `hybrid` (si el paso 8 no confirma la mejora, cambiar a `bm25`).
- [x] `sentence_transformers` solo se importa al cargar el modelo (`VectorIndex._load_model`): `bm25` no lo carga nunca.
- [x] Índice antiguo sin `vectors.npy` + `--retriever hybrid` → `IndexNotFoundError` ("Run 'index' first").
- [x] Tests de la CLI (`test_cli_parsing.py`, `TestSearch` de `test_main.py`): guardan solo BM25, así que usan `--retriever bm25`. `make lint` limpio y 1049 tests pasan (2026-10-06).
- [ ] (Paso 9) Los tests de `index` ahora cargan el modelo de embeddings real: `make test` pasa de ~57 s a ~86 s. Sustituirlo por un modelo falso (monkeypatch de `VectorIndex._load_model`).

## Paso 8 — Medir ✅

- [x] `make test_rag_base` con `hybrid` (por defecto), 2026-10-06, en casa: **idéntico al experimento**.

  | Dataset | @1 | @3 | @5 | @10 | BM25 @5 |
  |---|---|---|---|---|---|
  | docs | 0.640 | 0.870 | **0.910** | 0.920 | 0.870 |
  | code | 0.545 | 0.778 | **0.828** | 0.859 | 0.758 |

  Tiempos: `index` **19.6 s** (vectores ~4 s, 117 lotes); `search_dataset` **9.2 s** docs + **9.4 s** code (incluye arrancar el proceso y cargar el modelo de embeddings, ~6 s; cada pregunta ~0.01 s).
- [x] Comprobado contra el subject: `hybrid` **puede quedarse por defecto** (umbrales: indexado ≤ 5 min, 200 preguntas ≤ 90 s, recall@5 ≥ 0.80 docs / 0.50 code; el subject no obliga a que el retriever por defecto sea léxico). Salvedad: con `hybrid`, `index` necesita el modelo de embeddings en la caché o internet para descargarlo la primera vez (como Qwen para `answer`).
- [x] `make test_limits` (2026-10-06, en casa, `hybrid`): indexado **28.2 s** / 300 s ✅; recuperación **199 preguntas en 19.4 s** / 90 s ✅ (~13 preguntas/s). En el campus (~3×): ~85 s y ~58 s, dentro de los límites. Repetir allí si es posible.
- [x] Con `--retriever bm25` (2026-10-06): docs 0.610 / 0.830 / **0.870** / 0.920, code 0.515 / 0.717 / **0.758** / 0.818: los de siempre. `search_dataset` ~2 s frente a ~9 s con `hybrid` (la diferencia es cargar PyTorch y el modelo una vez; cada pregunta ~7 ms: encode 0.4, matmul 2, `maximum.at` 4).
- [ ] Batería (`make test_answer`) con `hybrid`: 04–10 vistas; 07 sale **al revés** (❌, antes 🟡) aunque los chunks de `kv_cache_coordinator.py` están arriba (sensibilidad de Qwen 0.6B a la mezcla de chunks). Falta ver 01–03. Batería 521 s frente a 190 s: el híbrido solo suma ~6 s por `answer`; repetir una pregunta con `--retriever bm25` para ver si el resto es memoria del PC.

## Paso 9 — Tests y documentación ✅

- [x] `tests/conftest.py`: fixture `autouse` que cambia `VectorIndex._load_model` por `FakeEncoder` (bolsa de palabras con hash fijo, huecos en caché, `np.add.at`) en **todos** los tests. `make test` ya no carga el modelo de embeddings: **1093 tests en ~60 s**.
- [x] `tests/test_vector_index.py`: ventanas (casos del paso 1), prefijo de la ruta y orden de `owners`, máximo por chunk (ejemplo del paso 5), guardar/cargar, errores de `load` (falta fichero, basura, formas incoherentes), carpeta no escribible.
- [x] `tests/test_hybrid_retriever.py`: fórmula (ejemplo del paso 6), `k`, `CANDIDATES`, `ALPHA = 0` → igual que BM25, `ALPHA = 1` → orden por similitud, solo candidatos de BM25, empates, sin resultados, índices que no coinciden.
- [x] Docstrings de `HybridRetriever` (y `scored_hits: list[SearchHit]`).
- [x] `make lint` y `make test` limpios. (`from conftest import FakeEncoder`, no `tests.conftest`: si no, mypy ve el fichero con dos nombres.)
- [x] README: descripción y estado, opciones (`--retriever`), arquitectura (diagrama + `VectorIndex` / `HybridRetriever`), Retrieval method (*Vector re-ranking*), resultados actuales (hybrid y bm25), tiempos, evolución, tabla de mejoras (✅), tests, decisiones de diseño.
- [x] DEVLOG: Step 21.

## Siguiente

- Bonus por orden recomendado: 4 (caché), 5 (API HTTP), 3 (indexado incremental). Antes de 4 y 5: capa de servicio y clase base `Retriever` (sustituye a `BM25Index | HybridRetriever`).
- Pendiente menor: el aviso "unauthenticated requests to the HF Hub" al cargar el modelo de embeddings (¿hace alguna petición aunque esté en caché?).

---

# Bonus 4 — Caché (plan del 2026-10-06)

Subject: *"Caching: cache the index and query results to speed up cold start and repeated queries."* Dos partes: **arranque en frío** (cargar menos y más rápido) y **preguntas repetidas** (no volver a buscar). Cuenta solo si funciona y se puede demostrar: medir antes y después.

## Dónde se va el tiempo hoy (medido)

| Qué | Tiempo | Cuándo |
|---|---|---|
| Arrancar `uv` + Python + importar `src.commands` | ~0.25 s | siempre |
| Cargar `index.json` (leer 0.45 + validar con pydantic 0.38) | ~0.8 s | siempre |
| Importar `sentence_transformers` + cargar `potion-base-8M` | **~6.5 s** | `hybrid`, primera pregunta |
| Cada pregunta (vectorizar + producto + máximo) | ~7 ms | `hybrid` |
| `answer`: Qwen | ~5.6 s de carga + ~20 s por respuesta | `answer` |

Lo que más pesa con diferencia es `sentence_transformers` (importa PyTorch). Prueba hecha: el modelo es **estático** (`tokenizer.json` + una tabla `(29528, 256)` en `model.safetensors`, y luego media + normalizar). Con solo `tokenizers` + numpy da **los mismos vectores** (diferencia máx. 1.5e-8) y carga en **0.65 s** en vez de 6.5 s.

## Fase 0 — Terminar `Retriever` ✅

- [x] `Retriever(ABC)` con `search` abstracto, en `src/base_retriever.py`. `BM25Index` y `HybridRetriever` heredan de él; `VectorIndex` no (no tiene `search`). `commands.py` usa `Retriever` en `_load_retriever` y `_search_question`.
- [x] `make lint-strict` limpio, 1093 tests.
- [ ] Test: `Retriever()` → `TypeError`; `BM25Index` y `HybridRetriever` son subclases (con los tests de la fase 1).

## Fase 1 — Arranque en frío: modelo de embeddings sin PyTorch ✅

Idea: la "caché del índice". `index` copia lo que el modelo necesita **dentro de `--index_dir`**, y las búsquedas lo leen de ahí con numpy, sin `sentence_transformers` ni la caché de Hugging Face.

- [x] 1.1 `StaticEncoder` (`src/static_encoder.py`): `from_pretrained` (caché primero, descarga solo si falta, `RagError` claro) y `encode(texts)` (media de `table[ids]`, normalizado, texto sin tokens → ceros). Comprobado con las 59 828 ventanas reales: **igual que `vectors.npy`** (diferencia máx. 9e-8). Carga en 0.27 s.
- [x] 1.2 `save` / `load`: `embeddings.npy` + `tokenizer.json`. `tokenizers` lanza `Exception` (error de Rust) al leer/escribir mal: capturado aparte. Comprobado: ida y vuelta idéntica; falta fichero → `IndexNotFoundError`; basura → `VectorIndexError`; tabla no 2D → `VectorIndexError`; carpeta no escribible → `PathError`.
- [x] 1.3 `VectorIndex` usa `StaticEncoder`: fuera `sentence_transformers` y `TYPE_CHECKING`; `_load_model` → `StaticEncoder.from_pretrained`; `build` por lotes de `BATCH_SIZE = 512` con tqdm ("Embedding windows"); `save` guarda también el modelo; `load` lo carga del índice.
- [x] Docstrings de `StaticEncoder` y `VectorIndex` al día.
- [x] Tests: `FakeEncoder` del `conftest` sin `normalize_embeddings` y con `save` (marca `fake_encoder.txt`; el `StaticEncoder.load` parcheado devuelve el falso solo en esas carpetas). Nuevo `tests/test_static_encoder.py` (23 tests) con un tokenizer `WordLevel` de 5 palabras y una tabla conocida: media, normalizado, `[UNK]`, texto vacío, guardar/cargar, errores, `from_pretrained` (caché, descarga, sin red), y `Retriever` (abstracta, subclases). **1116 tests**, `make lint-strict` limpio.
- [x] Medido sin tocar `data/processed` (copia del índice + modelo guardado al lado): `_load_retriever("hybrid")` + primera pregunta = **1.1–1.6 s** (antes ~7.3 s: 0.8 BM25 + 6.5 `sentence_transformers`). **PyTorch no se importa.**
- [x] 1.4 `uv remove sentence-transformers` (siguen `torch`/`transformers` para Qwen). Commit `0885414`.
- [x] 1.5 Medido con el índice real (2026-10-06, `make test_limits` + `evaluate`):

  | | Antes (`sentence-transformers`) | Ahora (`StaticEncoder`) |
  |---|---|---|
  | `index` | 28.2 s | **13.2 s** (vectorizar: 117 lotes en 4 s) |
  | 199 preguntas (2 procesos) | 19.4 s | **5.4 s** (82 y 68 preguntas/s) |
  | Recall@5 docs / code | 0.910 / 0.828 | **0.910 / 0.828** (todos los k idénticos) |

  `data/processed` ahora tiene `index.json`, `vectors.npy`, `owners.npy`, `embeddings.npy`, `tokenizer.json`.
- [x] ~~(Opcional) `encode` sin bucle~~: medido y descartado. Bucle 1.6 s; `np.add.reduceat` 13–16 s (copia las 7.3 M filas antes de sumar); matriz de conteos × tabla 1.6 s. El trabajo son las sumas, no las vueltas de Python. Tokenizar (2.4 s) pesa más.
- [x] README y DEVLOG (Step 22) al día.

## Fase 2 — Preguntas repetidas: `CachedRetriever` ✅

**Para qué:** si una pregunta ya se buscó, devolver el resultado guardado sin buscar; y si **todas** están en caché, no cargar ni los índices. Objetivo: repetir `search_dataset` pasa de ~3 s a unas décimas; `search` repetido, sin cargar nada.

**Diseño (patrón *decorator*):** `CachedRetriever` **es** un `Retriever` y **contiene** otro (BM25 o el híbrido), que no cambian. En vez del retriever ya cargado recibe **una función que sabe cargarlo** (`load_inner`), y solo la llama la primera vez que falta una pregunta.

Cuatro pasos: 2.1 la clase en memoria → 2.2 guardar en disco con la huella del índice → 2.3 CLI → 2.4 tests, medidas y docs.

### Paso 2.1 — La clase, solo en memoria ✅

Fichero `src/cached_retriever.py` (ahora se llama `cached_retrived.py`, renombrar). Esqueleto que se le dio:

```python
from collections.abc import Callable

from .base_retriever import Retriever
from .models import SearchHit


class CachedRetriever(Retriever):
    def __init__(self, load_inner: Callable[[], Retriever]) -> None:
        self._load_inner = load_inner              # la función, sin llamarla
        self._inner: Retriever | None = None       # el retriever, aún sin cargar
        self._entries: dict[str, list[SearchHit]] = {}

    def _retriever(self) -> Retriever:
        # si self._inner es None: self._inner = self._load_inner()
        # devolver self._inner   (misma idea que VectorIndex._encoder)

    @staticmethod
    def _key(query: str, k: int) -> str:
        # legible, distingue pregunta Y k: f"{k}:{query}"

    def search(self, query: str, k: int) -> list[SearchHit]:
        # key = self._key(query, k)
        # si no está: self._entries[key] = self._retriever().search(query, k)
        # devolver una COPIA: list(self._entries[key])
```

- `Callable[[], Retriever]` = función sin argumentos que devuelve un `Retriever`. En la CLI: `CachedRetriever(lambda: self._load_retriever(index_dir, retriever))` (el `lambda` no se ejecuta hasta que alguien lo llama).
- Devolver una copia: si quien llama cambia la lista, lo guardado no se estropea.
- No capturar errores aquí: un "Run 'index' first" al cargar tiene que llegar al usuario igual que ahora.

**Prueba que se le dio** (debe salir `created, loads = 0` y luego `loads = 1  searches = 3`):

```bash
uv run python -c "
from src.cached_retriever import CachedRetriever
from src.base_retriever import Retriever
loads, searches = [], []
class Fake(Retriever):
    def search(self, query, k):
        searches.append(query); return []
def load():
    loads.append(1); return Fake()
c = CachedRetriever(load)
print('created, loads =', len(loads))
c.search('lora', 5); c.search('lora', 5); c.search('lora', 10); c.search('cuda', 5)
print('loads =', len(loads), ' searches =', len(searches))
"
```

- [x] `CachedRetriever` en memoria; la prueba da `0` / `1` / `3` (2026-10-06). `mypy --strict` limpio. Fichero ya renombrado a `src/cached_retriever.py`; comprobar que tiene salto de línea final (flake8 W292).
- [ ] `make lint-strict` limpio (docstrings al final de la fase, como siempre).

### Paso 2.2 — Guardar en disco, con la huella del índice ✅

**Para qué:** que la caché sobreviva entre comandos (cada comando de la CLI es un proceso nuevo) sin devolver **nunca** resultados de un índice viejo o de otra configuración.

- [ ] **`namespace`** (un `str`) que identifica "qué produjo estos resultados". Se construye en la CLI (paso 2.3) y se pasa a `CachedRetriever`:
  - el retriever (`bm25` / `hybrid`) y sus parámetros: `HybridRetriever.CANDIDATES`, `ALPHA`, `VectorIndex.MODEL_NAME`;
  - la **huella del índice**: `st_size` y `st_mtime_ns` de `index.json` y `vectors.npy` (`Path.stat()`, no hay que leer los ficheros: es instantáneo). Reindexar cambia `st_mtime_ns` → caché nueva.
  - Si falta un fichero del índice, **no** fallar al calcular la huella: dejar que el error salga al cargar el retriever ("Run 'index' first"), como ahora. Por ejemplo, huella `"missing"` para ese fichero.
- [ ] **Modelo pydantic** en `models.py`: `QueryCache(namespace: str, entries: dict[str, list[SearchHit]])` (el proyecto valida todo lo que lee de disco con pydantic; **no** `pickle`).
- [ ] `CachedRetriever.__init__(load_inner, cache_path, namespace)`: al crearse, **intenta** leer `cache_path` con `QueryCache.model_validate_json`:
  - no existe, `namespace` distinto, JSON roto o no válido → empezar con `{}`; **nunca un error** (la caché es una ayuda, no un requisito);
  - ok → `self._entries = data.entries`.
- [ ] `save()`: escribe `QueryCache(namespace, entries)` solo si se añadió algo (un atributo `self._changed`). **Atómico:** escribir en `cache_path.with_suffix(".tmp")` y luego `os.replace(tmp, cache_path)` (si el proceso muere a medias, el fichero viejo sigue entero). Si no se puede escribir (`OSError`) → aviso en `stderr` y seguir: el comando ya hizo su trabajo.
- [ ] Ubicación: `index_dir / "query_cache.json"` (constante `FILE_NAME` en la clase).
- [ ] Tamaño: guardar el `SearchHit` entero (con el texto del chunk) es lo más simple: ~2 KB por hit → ~4 MB para 200 preguntas con k = 10. Medirlo; si molesta, guardar solo `chunk_id` + `score` (pero entonces la caché necesita los chunks de BM25 para rehacer los hits, y eso obliga a cargar el índice: se pierde la ventaja). **Recomendado: el `SearchHit` entero.**

Esqueleto que se le dio (2.2):

```python
# models.py  (HECHO el esqueleto; falta docstring)
class QueryCache(BaseModel):
    namespace: str
    entries: dict[str, list[SearchHit]]

# cached_retriever.py
FILE_NAME = "query_cache.json"

def __init__(self, load_inner: Callable[[], Retriever],
             cache_path: Path, namespace: str) -> None:
    # guardar los tres en atributos
    self._entries = self._read()      # lo que haya en el fichero, o {}
    self._changed = False             # ¿se añadió algo desde que se leyó?

def _read(self) -> dict[str, list[SearchHit]]:
    try:
        text = self._cache_path.read_text(encoding="utf-8")
        data = QueryCache.model_validate_json(text)
    except (OSError, ValidationError):     # no existe, ilegible, JSON roto
        return {}
    # otro namespace → {} ; el mismo → data.entries

# en search, al guardar: self._changed = True

def save(self) -> None:
    if not self._changed:
        return
    data = QueryCache(namespace=self._namespace, entries=self._entries)
    tmp = self._cache_path.with_suffix(".tmp")
    # escribir data.model_dump_json() en tmp, luego os.replace(tmp, self._cache_path)
    # OSError → aviso en stderr y seguir (sin excepción)
```

Por qué `.tmp` + `os.replace`: si el proceso se corta escribiendo, queda el fichero viejo entero o el nuevo entero, nunca uno a medias.

**Prueba para el usuario** (dársela cuando termine 2.2): crear un `CachedRetriever` con el `Fake` de 2.1, buscar, `save()`, crear **otro** con el mismo `namespace` → la búsqueda no llama a `load` (`loads = 0`); con otro `namespace` → sí; fichero con basura → empieza vacía, sin error.

### Paso 2.3 — Conectarlo a la CLI

- [ ] `RagCLI._load_retriever(index_dir, retriever, cache)`:
  - `cache=False` → como ahora;
  - `cache=True` → `CachedRetriever(lambda: <lo de ahora>, Path(index_dir) / CachedRetriever.FILE_NAME, namespace)`.
  - La función que construye el `namespace` (huella + parámetros) puede ser un `@staticmethod` de `CachedRetriever` o de `RagCLI`; mejor en `CachedRetriever` (`fingerprint(index_dir, retriever)`), así la CLI no sabe cómo se hace.
- [ ] `--cache` (`bool`, por defecto `True`) en `search`, `search_dataset` y `answer`, con su línea en `Args` del docstring (sale en `--help`). Para la demo: `--cache False`. Ojo con Fire y los booleanos: `--cache=False` o `--nocache` (ver README → *Every option*: los flags van **después** de los argumentos).
- [ ] Llamar a `save()` al final de cada comando (después del bucle de `search_dataset`, no por pregunta). Como `_load_retriever` devuelve un `Retriever`, guardar el `CachedRetriever` en una variable con su tipo, o hacer que `save` sea un método de `Retriever` que no hace nada por defecto (más simple en los comandos; decidir con el usuario).
- [ ] `index` borra `query_cache.json` si existe (la huella ya la invalida, pero así no se queda un fichero viejo de varios MB).
- [ ] `make cleandata` ya borra `data/processed/*`: comprobar que incluye la caché.

### Paso 2.4 — Tests, medidas y documentación

- [ ] `tests/test_cached_retriever.py` con un retriever falso que cuenta cargas y búsquedas:
  - hit / miss; el interno **no se carga** si todo está en caché; otra `k` = otra entrada; devuelve copias;
  - guardar → otra instancia lo lee; otro `namespace` → vacía; JSON roto → vacía; carpeta no escribible → aviso, sin excepción; no escribe si no cambió nada;
  - la huella cambia si cambia `index.json` (`os.utime` para tocar la fecha);
  - CLI: `search` dos veces con `--retriever bm25` y la segunda no carga el índice (parchear `BM25Index.load` para contar).
- [ ] **Medir** (el usuario ejecuta los comandos): `search_dataset` docs con caché fría y caliente; `search` repetido; `--cache False`; `index` de nuevo → la siguiente búsqueda no usa la caché vieja. Comprobar que el JSON de resultados es **idéntico** con y sin caché (`diff`) y que el recall no cambia (0.910 / 0.828).
- [ ] `make test_limits` sigue pasando (la primera vez es sin caché: `index` la borra).
- [ ] Docstrings de todo `cached_retriever.py`; `make lint-strict`.
- [ ] README (arquitectura, opción `--cache`, tiempos antes/después, tabla de mejoras, tests, decisiones) y DEVLOG (Step 23).

## Fase 3 (opcional) — Caché de respuestas de `answer`

La pregunta repetida que más cuesta es `answer` (~25 s). Misma idea: clave = pregunta + `k` + retriever + huella + versión del prompt (`MAX_CONTEXT_TOKENS`, `MAX_NEW_TOKENS`, `Generator.MODEL_NAME`). Hit → no se carga PyTorch ni Qwen: ~25 s → < 1 s. Decidir después de la fase 2 si vale la pena (el subject habla de *query results*; esto es un extra fácil de demostrar).

## Fase 4 — Demostración, medidas y documentación

- [ ] Regla `make test_cache`: `search_dataset` dos veces (fría y caliente) y `search` repetido, imprimiendo los tiempos.
- [ ] README: arquitectura (`StaticEncoder`, `CachedRetriever`, ficheros nuevos en `data/processed`), opción `--cache`, tabla de tiempos antes/después, tabla de mejoras, decisiones (decorator, huella del índice, caché que nunca rompe el comando).
- [ ] DEVLOG: Step 22.
- [ ] `make lint-strict`, `make test`, `make test_limits`, `make test_rag_base` (0.910 / 0.828) limpios.

## Qué NO hacer

- `pickle` para la caché o el índice: cargar un pickle ejecuta código; el proyecto ya eligió JSON + pydantic por eso.
- `functools.lru_cache` en `search`: solo vive dentro de un proceso, y cada comando de la CLI es un proceso nuevo (servirá para el bonus 5, la API, que sí se queda abierta: `CachedRetriever` con su diccionario en memoria ya lo cubre).
- Cambiar el formato de `index.json` para ganar 0.8 s: con la fase 2, una pregunta en caché ni siquiera lo carga.

---

# Bonus 5 — API HTTP local (plan del 2026-10-06)

Subject: *"Local HTTP API: expose querying the index and answering questions over a small local HTTP API, so the system can be driven by something other than the CLI."* Cuenta solo si funciona y se puede demostrar.

## Decisión: FastAPI + uvicorn (2026-10-06)

Elegido por el usuario frente a `http.server` (librería estándar). Motivos:
- **Encaja con el código que ya hay:** FastAPI valida la entrada y serializa la salida con los modelos pydantic del proyecto (`StudentSearchResults`, `StudentSearchResultsAndAnswer`): las respuestas tienen exactamente el mismo formato que la CLI, sin escribir JSON a mano.
- **Menos código propio y menos sitios donde equivocarse:** rutas, parseo, validación (422 automático) y códigos de error vienen hechos; con `http.server` habría que escribirlos.
- **Demostrable:** `/docs` (Swagger) permite probar la API desde el navegador en la defensa.
- Coste: dos dependencias (`fastapi`, `uvicorn`); el subject permite cualquier librería. Para los tests, `fastapi.testclient` (necesita `httpx`, dependencia de desarrollo).

## Diseño

- Clase `RagApi` en `src/api.py` (100 % OOP): guarda el retriever y el `Generator`, y registra sus métodos como rutas sobre un `FastAPI` en `__init__`.
- Comando `serve --host 127.0.0.1 --port 8000 --index_dir --retriever --cache` en `RagCLI`: crea `RagApi` y llama a `uvicorn.run`. **Local:** por defecto solo `127.0.0.1`.
- El servidor es **un proceso que se queda abierto**: el retriever se carga una vez (con `CachedRetriever` en memoria entre peticiones) y Qwen se carga en la **primera** petición a `/answer`, no al arrancar.
- Mismas reglas que la CLI: nunca un traceback; `RagError` → respuesta HTTP con su mensaje.

## Pasos

### 5.1 — Dependencias y esqueleto que arranca ✅
- [x] `uv add fastapi uvicorn` y `uv add --dev httpx`.
- [x] `src/api.py`: `RagApi` con `GET /health` → `{"status": "ok"}`.
- [x] Comando `serve` en `RagCLI` (`uvicorn.run(api.app, host=..., port=...)`).
- [x] Prueba: `uv run -m src serve` + `curl http://127.0.0.1:8000/health`; `/docs` en el navegador.

### 5.2 — `GET /search` ✅
- [x] Parámetros `query` (no vacía, ni solo espacios) y `k` (> 0) validados por FastAPI; reutilizar `RagCLI._search_question`. Respuesta: `StudentSearchResults`.
- [x] El retriever se crea al arrancar con `_load_retriever` (con caché, no carga nada hasta la primera pregunta nueva).
- [x] `flush()` tras cada búsqueda (solo escribe si la pregunta era nueva).
- [x] Docstrings y `description` de la app.

### 5.3 — `GET /answer` ✅ (GET y no POST, como `/search`)
- [x] Parámetros `query` y `k`; respuesta `StudentSearchResultsAndAnswer`. Sin fuentes → `NO_SOURCES` sin cargar Qwen.
- [x] `Generator` cargado una vez, la primera vez que hace falta, con el `dtype` de `serve`.
- [x] `threading.Lock` alrededor de la carga y la generación, y otro alrededor de search + flush.

### 5.4 — Errores ✅
- [x] `RagError` → 503 `{"detail": "<mensaje>"}` con un manejador global. Nunca un 500 con traceback.
- [x] Entrada mala (vacía, solo espacios, `k` ≤ 0 o no entero) → 422.
- [x] Puerto ocupado → exit 3 de uvicorn: se deja así.

### 5.5 — Tests, medidas y documentación ✅
- [x] `tests/test_api.py` con `TestClient` (36 tests, sin Qwen ni PyTorch: `Generator` falso y `src.generator` sustituido en `sys.modules`).
- [x] README (estado, `serve`, sección *HTTP API* con `curl`, tests, decisiones, limitaciones) y DEVLOG (Steps 24–25).
- [x] `make serve`.

### Siguiente
- [ ] Commit (lo hace el usuario).
- [ ] Practicar la demo: `make serve`, `/docs` en el navegador, `curl` desde otra terminal (`/search` y `/answer`), y enseñar un error (503 sin índice, 422 con `k=0`).
- [ ] Bonus 3: indexado incremental.
