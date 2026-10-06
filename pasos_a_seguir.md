# Pasos a seguir: índice de vectores y búsqueda híbrida (bonus 1 y 2)

Objetivo medido (README → *Semantic embeddings: tuning `potion-base-8M`*):
recall@5 **docs 0.870 → 0.910** y **code 0.758 → 0.828**, con unos 3–8 s más de indexado.

Configuración elegida:

| Qué | Valor |
|---|---|
| Modelo | `minishlab/potion-base-8M` (Model2Vec, 30 MB, se carga con `sentence-transformers`) |
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
- **No ejecutar Qwen** salvo que lo pida (tarda ~20 s por pregunta). El modelo de embeddings sí se puede ejecutar: es rápido.
- Medir cada cambio por separado y documentarlo (README → Performance, DEVLOG).

### Estado del proyecto (2026-10-05)

- Parte obligatoria **completa** (commit "Ended the mandatory part of the proyect."). Recall@5 con BM25: docs **0.870**, code **0.758**. `make test` (1049 tests, ~20 s, nunca carga Qwen) y `make lint` limpios antes de empezar los bonus.
- Bonus en curso: **1 (embeddings) y 2 (híbrido)**. Después, por orden recomendado: 4 (caché), 5 (API HTTP), 3 (indexado incremental, el más caro).
- Antes de los bonus 4 y 5 se recomendó sacar la lógica de los comandos a una capa de servicio y crear una clase base `Retriever`.
- `src/vector_index.py`: pasos 1–3 hechos (ventanas, prefijo de ruta, `build`, carga del modelo sin red). `save`/`load` (paso 4), `chunk_score` (paso 5) y `HybridRetriever` en `src/hybrid_retriever.py` (paso 6) hechos, y conectados a la CLI (paso 7). Paso 8: recall medido (docs 0.910, code 0.828 @5); `test_limits` pasa; falta la comparación con `bm25`.
- `data/processed/vectors.npy` es una prueba anterior, **no** el formato final (una fila por chunk, sin ventanas). Se puede sobrescribir.
- `test_embedding.py` (raíz) es un script de prueba del usuario; usa `minishlab/potion-base-8M`.
- Modelos en la caché de Hugging Face: solo `minishlab/potion-base-8M` y `Qwen/Qwen3-0.6B` (los demás probados se borraron).

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

## Fase 1 — Arranque en frío: modelo de embeddings sin PyTorch ✅ (falta quitar la dependencia)

Idea: la "caché del índice". `index` copia lo que el modelo necesita **dentro de `--index_dir`**, y las búsquedas lo leen de ahí con numpy, sin `sentence_transformers` ni la caché de Hugging Face.

- [x] 1.1 `StaticEncoder` (`src/static_encoder.py`): `from_pretrained` (caché primero, descarga solo si falta, `RagError` claro) y `encode(texts)` (media de `table[ids]`, normalizado, texto sin tokens → ceros). Comprobado con las 59 828 ventanas reales: **igual que `vectors.npy`** (diferencia máx. 9e-8). Carga en 0.27 s.
- [x] 1.2 `save` / `load`: `embeddings.npy` + `tokenizer.json`. `tokenizers` lanza `Exception` (error de Rust) al leer/escribir mal: capturado aparte. Comprobado: ida y vuelta idéntica; falta fichero → `IndexNotFoundError`; basura → `VectorIndexError`; tabla no 2D → `VectorIndexError`; carpeta no escribible → `PathError`.
- [x] 1.3 `VectorIndex` usa `StaticEncoder`: fuera `sentence_transformers` y `TYPE_CHECKING`; `_load_model` → `StaticEncoder.from_pretrained`; `build` por lotes de `BATCH_SIZE = 512` con tqdm ("Embedding windows"); `save` guarda también el modelo; `load` lo carga del índice.
- [x] Docstrings de `StaticEncoder` y `VectorIndex` al día.
- [x] Tests: `FakeEncoder` del `conftest` sin `normalize_embeddings` y con `save` (marca `fake_encoder.txt`; el `StaticEncoder.load` parcheado devuelve el falso solo en esas carpetas). Nuevo `tests/test_static_encoder.py` (23 tests) con un tokenizer `WordLevel` de 5 palabras y una tabla conocida: media, normalizado, `[UNK]`, texto vacío, guardar/cargar, errores, `from_pretrained` (caché, descarga, sin red), y `Retriever` (abstracta, subclases). **1116 tests**, `make lint-strict` limpio.
- [x] Medido sin tocar `data/processed` (copia del índice + modelo guardado al lado): `_load_retriever("hybrid")` + primera pregunta = **1.1–1.6 s** (antes ~7.3 s: 0.8 BM25 + 6.5 `sentence_transformers`). **PyTorch no se importa.**
- [ ] 1.4 Dependencias: `sentence-transformers` ya no se usa en `src/` ni en `tests/` → `uv remove sentence-transformers` (sigue haciendo falta `torch`/`transformers` para Qwen).
- [x] 1.5 Medido con el índice real (2026-10-06, `make test_limits` + `evaluate`):

  | | Antes (`sentence-transformers`) | Ahora (`StaticEncoder`) |
  |---|---|---|
  | `index` | 28.2 s | **13.2 s** (vectorizar: 117 lotes en 4 s) |
  | 199 preguntas (2 procesos) | 19.4 s | **5.4 s** (82 y 68 preguntas/s) |
  | Recall@5 docs / code | 0.910 / 0.828 | **0.910 / 0.828** (todos los k idénticos) |

  `data/processed` ahora tiene `index.json`, `vectors.npy`, `owners.npy`, `embeddings.npy`, `tokenizer.json`.
- [x] ~~(Opcional) `encode` sin bucle~~: no hace falta, vectorizar el corpus dentro de `index` tarda 4 s (los 7.5 s medidos antes eran con la máquina cargada).

## Fase 2 — Preguntas repetidas: `CachedRetriever`

Un `Retriever` que envuelve a otro (patrón *decorator*): BM25 y el híbrido no cambian.

- [ ] `CachedRetriever(Retriever)` en `src/cached_retriever.py`:
  - `__init__(load_inner, cache_path, namespace)`: `load_inner` es una función que carga el retriever de verdad **solo cuando hace falta** → si todas las preguntas están en caché, **no se carga ningún índice ni modelo**;
  - `search(query, k)`: clave = `sha256(namespace + k + query)`; si está, devuelve los hits guardados; si no, carga el interno (una vez), busca, guarda.
- [ ] `namespace` = qué retriever + sus parámetros (`CANDIDATES`, `ALPHA`, `MODEL_NAME`) + la **huella del índice** (`st_size` y `st_mtime_ns` de `index.json` y `vectors.npy`). Reindexar o cambiar α invalida la caché sola: nunca resultados viejos.
- [ ] En disco: `index_dir/query_cache.json`, modelo pydantic `QueryCache(namespace: str, entries: dict[str, list[SearchHit]])`. Al abrir: no existe, otro `namespace` o fichero roto → empezar vacía (**nunca** un error: la caché es una ayuda, no un requisito). Guardar al final del comando (no por pregunta) y de forma atómica (fichero temporal + `os.replace`). `index` la borra.
- [ ] Tamaño: guardar solo `chunk_id` + `score` y rehacer los `SearchHit` con los chunks de BM25 (más pequeño), **o** el `SearchHit` entero (más simple, ~4 MB por 200 preguntas). Empezar por lo simple y medir.
- [ ] CLI: `_load_retriever` devuelve `CachedRetriever(...)`. Opción `--cache` (`True` por defecto) en `search`, `search_dataset` y `answer` para poder enseñar la diferencia (`--cache False`). Documentar en `--help`.
- [ ] **Comprobar**:
  - resultados con caché == sin caché (mismo JSON de `search_dataset`, mismo recall);
  - `search_dataset` docs: 1.ª vez ~9 s, 2.ª vez objetivo < 1 s; `search` repetido: sin cargar índices;
  - `index` de nuevo → la caché no se usa (cambió la huella);
  - `make test_limits` sigue pasando la primera vez (sin caché).
- [ ] Tests: hit/miss, el interno no se carga si todo está en caché, otra `k` u otro retriever = otra entrada, huella distinta → caché vacía, fichero roto → caché vacía, guardar/cargar, carpeta no escribible → avisa pero el comando funciona.

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
