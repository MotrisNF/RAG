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
- `src/vector_index.py`: pasos 1–3 hechos (ventanas, prefijo de ruta, `build`, carga del modelo sin red). Faltan `save`/`load`, `chunk_scores` y `HybridRetriever`.
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

## Paso 4 — Guardar y cargar

Métodos que quedan en `VectorIndex`: `save` y `load` (este paso) y `chunk_scores` (paso 5). La reordenación híbrida va en **otra clase** (paso 6).

- [ ] `save(index_dir)`: `np.save` de `vectors.npy` y `owners.npy` en la misma carpeta que `index.json`. Capturar `OSError` → `PathError`, como `BM25Index.save`.
- [ ] `load(index_dir)` (`@classmethod`): `np.load` de los dos. Errores claros:
  - falta algún fichero → `IndexNotFoundError` ("run `index` first");
  - no se puede leer → `PathError`;
  - formas incoherentes (`len(vectors) != len(owners)`, no es 2D…) → `DatasetError` o un error propio.
- [ ] Constantes con los nombres de fichero (`VECTORS_FILE`, `OWNERS_FILE`), como `FILE_NAME` en `BM25Index`.
- [ ] **Comprobar**: guardar y cargar → `np.array_equal` en los dos arrays.

## Paso 5 — Puntuar los chunks con una pregunta

- [x] Dónde vive el modelo: `_load_model()` (de la caché local primero, sin red) y `_encoder()` (lo carga la primera vez y lo guarda en `self._model`). `build` deja el modelo ya cargado en el índice.
- [ ] `chunk_scores(query) -> np.ndarray`:
  1. `query_vector = model.encode([query], normalize_embeddings=True)[0]` (pregunta **tal cual**).
  2. `scores = self._vectors @ query_vector` (una puntuación por ventana).
  3. `best = np.full(n_chunks, -1.0)` y `np.maximum.at(best, self._owners, scores)` (una por chunk).
- [ ] `n_chunks` = `self._owners.max() + 1` (o guardarlo).
- [ ] **Comprobar**: `best` tiene una puntuación por chunk, entre −1 y 1.

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

## Paso 6 — Reordenar los resultados de BM25 (bonus 2)

- [ ] Una clase aparte, `HybridRetriever` (en `vector_index.py` o en su propio `src/hybrid_retriever.py`), **no** dentro de `VectorIndex`: necesita los dos índices, y así `VectorIndex` no depende de `BM25Index`. Cada clase hace una cosa: BM25 busca por palabras, `VectorIndex` puntúa por significado y `HybridRetriever` combina.
- [ ] `__init__(bm25, vectors)`: guarda los dos índices.
- [ ] Comprobar que coinciden: el mayor número de `owners` debe ser el número de chunks de BM25 menos uno. Si no, uno de los dos se reconstruyó sin el otro → error claro ("run `index` again").
- [ ] Constantes de clase: `CANDIDATES = 30`, `ALPHA = 0.5` (comentario que remita al README).
- [ ] `search(query, k) -> list[SearchHit]`, la **misma** firma que `BM25Index.search`, para que los comandos usen uno u otro sin más cambios:
  1. `hits = bm25.search(query, CANDIDATES)`.
  2. Para cada candidato: `(1 − ALPHA) × hit.score / mejor score BM25 + ALPHA × best[id del chunk]`.
  3. Ordenar de mayor a menor y devolver los `k` primeros como `SearchHit` (con la puntuación combinada).
- [ ] El id de cada chunk: su posición en la lista de chunks del índice BM25 (los dos índices se construyen con la **misma** lista y en el mismo orden).
- [ ] Sin resultados de BM25 → lista vacía (como ahora).
- [ ] **Comprobar**: con `ALPHA = 0` el orden es **exactamente** el de BM25.

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
best = vectors.chunk_scores(query)             # una puntuación por chunk
top_bm25 = hits[0].score                       # la mayor (la primera)
combined = [(1 - ALPHA) * hit.score / top_bm25 + ALPHA * best[id del chunk] for hit in hits]
```

y ordenar los `hits` por `combined`, de mayor a menor, quedándose con los `k` primeros. El "id del chunk" es su posición en la lista de chunks de BM25 (paso 6, punto anterior).

**Ojo: hoy `SearchHit` no lleva el id del chunk** (solo `chunk` y `score`), y `BM25Index.search` lo tiene dentro (`chunk_id`) pero no lo devuelve. Opciones:
- añadir un campo `chunk_id: int` a `SearchHit` y rellenarlo en `BM25Index.search` (lo más limpio: un dato que ya existe, sin buscarlo después);
- o, en `HybridRetriever`, un diccionario `{id(chunk): posición}` hecho una vez en `__init__` (funciona, pero depende de que sean los mismos objetos).

## Paso 7 — Conectarlo a la CLI

- [ ] `index`: después de BM25, construir y guardar el `VectorIndex` en la misma `--index_dir`.
- [ ] `search`, `search_dataset`, `answer`: opción `--retriever` con `Literal["bm25", "hybrid"]` (un tipo compartido en `models.py`, como `DtypeMode`, para no importar torch en la CLI).
- [ ] Por defecto: `hybrid` si el paso 8 confirma la mejora.
- [ ] `sentence_transformers` solo se importa cuando se usa `hybrid` (carga perezosa).
- [ ] Índice antiguo sin vectores + `--retriever hybrid` → error claro ("run `index` again").

## Paso 8 — Medir

- [ ] `make test_rag_base` (o moulinette) con `--retriever hybrid`: debería dar **docs ≈ 0.91, code ≈ 0.83** en recall@5. Si sale distinto, algún paso no coincide con lo medido.
- [ ] `make test_limits`: indexado muy por debajo de 5 min y 200 preguntas por debajo de 90 s.
- [ ] Con `--retriever bm25`: los números de siempre (0.870 / 0.758).

## Paso 9 — Tests y documentación

- [ ] Tests **sin el modelo real**: un objeto falso con un método `encode` que devuelva vectores inventados.
  - ventanas (casos del paso 1);
  - prefijo de la ruta y orden de `owners`;
  - máximo por chunk;
  - `ALPHA = 0` → igual que BM25;
  - guardar y cargar;
  - errores de `load` (falta fichero, formas incoherentes).
- [ ] Docstrings en todo `vector_index.py`.
- [ ] `make lint` y `make test` limpios.
- [ ] README: System architecture (componente nuevo), Retrieval method (reordenación), opciones de la CLI (`--retriever`), cambiar el "(to implement)" de la tabla de mejoras por ✅ con los números reales.
- [ ] DEVLOG: un Step nuevo con lo hecho y lo medido.
