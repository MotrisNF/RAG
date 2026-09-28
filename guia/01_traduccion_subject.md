# RAG against the machine

*¿Responderás a mis preguntas?*

> **Resumen:** Generación Aumentada por Recuperación (Retrieval-Augmented Generation), eso es todo. Ese es el objetivo de este proyecto.
>
> Versión: 2.0

> Nota del traductor: se han mantenido en inglés los nombres de modelos pydantic, campos, comandos, argumentos de la CLI, rutas, herramientas y cualquier término que tengas que usar literalmente en el desarrollo.

---

## Índice

- I. Preámbulo
- II. Prólogo
- III. Instrucciones sobre IA
- IV. Introducción
- V. Instrucciones comunes
  - V.1 Reglas generales
  - V.2 Makefile
  - V.3 Directrices adicionales
  - V.4 Requisitos adicionales
- VI. Parte obligatoria
  - VI.1 Indexar el código fuente
  - VI.2 Recuperación (retrieval)
  - VI.3 Generación de respuestas
  - VI.4 Modelos de datos
  - VI.5 Salida
  - VI.6 Interfaz de línea de comandos
  - VI.7 Recorrido completo de principio a fin
    - VI.7.1 Estructura de proyecto esperada para la evaluación
    - VI.7.2 Ejecutar el pipeline completo
- VII. Evaluación
  - VII.1 Métricas de evaluación
    - VII.1.1 Cálculo de Recall@k
    - VII.1.2 Rendimiento
- VIII. Requisitos del README
- IX. Parte bonus
- X. Entrega y evaluación entre pares
  - X.1 Instrucciones de recodificación

---

## Capítulo I — Preámbulo

El pipeline que vas a construir: ingerir e indexar un código fuente, recuperar los fragmentos más relevantes para una pregunta, entregárselos a un modelo local pequeño para que genere una respuesta fundamentada, y medir la calidad de la recuperación con **recall@k**.

---

## Capítulo II — Prólogo

Un modelo de lenguaje está congelado en el tiempo. Todo lo que "sabe" quedó fijado el día en que terminó su entrenamiento. Pregúntale por una librería publicada el mes pasado, por un código privado o por el informe de incidencias de ayer, y o bien se encogerá de hombros o se inventará una respuesta segura de sí misma sacada de la nada. Reentrenar un modelo cada vez que el mundo cambia es absurdamente caro, así que hacemos trampa. En lugar de meterle más conocimiento al modelo, le dejamos ir a buscar la información correcta en el momento de responder, igual que tú no memorizas un manual entero pero sabes qué página abrir.

Suena fácil hasta que lo intentas. El "manual" aquí es un código fuente completo: miles de archivos, cientos de miles de líneas. Encontrar los dos o tres fragmentos que realmente responden a una pregunta, y solo esos, es buscar una aguja en un pajar. Si buscas demasiado poco, se te escapa la respuesta; si buscas demasiado, ahogas al modelo en ruido.

Nuestra intuición es mala a esta escala, pero la buena ingeniería no. Recuperar la evidencia correcta de una montaña de datos y después hacer que un modelo pequeño responda fielmente a partir de ella: de eso trata este proyecto.

---

## Capítulo III — Instrucciones sobre IA

### ● Contexto

Durante tu aprendizaje, la IA puede ayudarte con muchas tareas distintas. Tómate el tiempo de explorar las capacidades de las herramientas de IA y cómo pueden apoyar tu trabajo. Sin embargo, abórdalas siempre con precaución y evalúa críticamente los resultados. Ya sea código, documentación, ideas o explicaciones técnicas, nunca puedes estar completamente seguro de que tu pregunta estuviera bien formulada o de que el contenido generado sea correcto. Tus compañeros son un recurso valioso para ayudarte a evitar errores y puntos ciegos.

### ● Mensaje principal

- ☛ Usa la IA para reducir tareas repetitivas o tediosas.
- ☛ Desarrolla habilidades de prompting —tanto de programación como no— que te servirán en tu carrera.
- ☛ Aprende cómo funcionan los sistemas de IA para anticipar y evitar riesgos, sesgos y problemas éticos comunes.
- ☛ Sigue desarrollando habilidades técnicas y habilidades transversales trabajando con tus compañeros.
- ☛ Usa solo contenido generado por IA que entiendas completamente y del que puedas responsabilizarte.

### ● Reglas para el estudiante

- Debes dedicar tiempo a explorar las herramientas de IA y entender cómo funcionan, para poder usarlas de forma ética y reducir posibles sesgos.
- Debes reflexionar sobre tu problema antes de escribir el prompt: esto te ayuda a escribir prompts más claros, detallados y relevantes, usando un vocabulario preciso.
- Debes adquirir el hábito de comprobar, revisar, cuestionar y probar sistemáticamente todo lo generado por IA.
- Debes buscar siempre la revisión de tus compañeros: no confíes únicamente en tu propia validación.

### ● Resultados de esta fase

- Desarrollar habilidades de prompting tanto generales como específicas de un dominio.
- Aumentar tu productividad con un uso eficaz de las herramientas de IA.
- Seguir reforzando el pensamiento computacional, la resolución de problemas, la adaptabilidad y la colaboración.

### ● Comentarios y ejemplos

- Te encontrarás a menudo en situaciones —exámenes, evaluaciones y más— en las que tendrás que demostrar una comprensión real. Prepárate y sigue desarrollando tanto tus habilidades técnicas como interpersonales.
- Explicar tu razonamiento y debatir con compañeros suele revelar lagunas en tu comprensión. Prioriza el aprendizaje entre pares.
- Las herramientas de IA a menudo carecen de tu contexto específico y tienden a dar respuestas genéricas. Tus compañeros, que comparten tu entorno, pueden aportar ideas más relevantes y precisas.
- Mientras que la IA tiende a generar la respuesta más probable, tus compañeros pueden aportar perspectivas alternativas y matices valiosos. Apóyate en ellos como control de calidad.

**✓ Buena práctica:**
Le pregunto a la IA: "¿Cómo pruebo una función de ordenación?". Me da algunas ideas. Las pruebo y reviso los resultados con un compañero. Refinamos el enfoque juntos.

**✗ Mala práctica:**
Le pido a la IA que escriba una función entera y la copio y pego en mi proyecto. Durante la evaluación entre pares no soy capaz de explicar qué hace ni por qué. Pierdo credibilidad y suspendo el proyecto.

**✓ Buena práctica:**
Uso la IA para ayudarme a diseñar un parser. Después repaso la lógica con un compañero. Detectamos dos bugs y lo reescribimos juntos: mejor, más limpio y totalmente comprendido.

**✗ Mala práctica:**
Dejo que Copilot genere el código de una parte clave de mi proyecto. Compila, pero no sé explicar cómo gestiona los pipes. Durante la evaluación no puedo justificarlo y suspendo el proyecto.

---

## Capítulo IV — Introducción

Ya vimos el *function calling* en **call_me_maybe**; esta vez el tema es **RAG**. Antes de ver qué es RAG, céntrate en lo que hace.

Un modelo solo "sabe" aquello con lo que fue entrenado, y reentrenarlo para añadir conocimiento es lento y costoso. RAG toma otro camino: en lugar de meter el conocimiento dentro del modelo, le das acceso a una fuente externa de tu elección y dejas que extraiga información de ella en el momento de responder.

En la práctica, RAG tiene cuatro etapas:

- **Indexación (Indexing):** organizar los datos para que se puedan buscar.
- **Recuperación (Retrieving):** comparar una pregunta con el índice y extraer los fragmentos más relevantes.
- **Aumento (Augmenting):** filtrar esos fragmentos y colocarlos en la ventana de contexto del modelo.
- **Generación (Generating):** leer ese contexto y producir la respuesta.

> Lee el documento entero antes de escribir código: cada etapa alimenta a la siguiente.

---

## Capítulo V — Instrucciones comunes

### V.1 Reglas generales

- Tu proyecto debe estar escrito en **Python 3.10 o superior**.
- Tu proyecto debe cumplir el estándar de código **flake8**.
- Tus funciones deben gestionar las excepciones con elegancia para evitar cierres inesperados. Usa bloques `try-except` para manejar posibles errores. Prefiere los *context managers* para recursos como archivos o conexiones, para asegurar la liberación automática. **Si tu programa se cierra por una excepción no gestionada durante la revisión, se considerará no funcional.**
- Todos los recursos (p. ej. descriptores de archivo, conexiones de red) deben gestionarse correctamente para evitar fugas. Usa *context managers* siempre que sea posible.
- Tu código debe incluir **type hints** para los parámetros de funciones, tipos de retorno y variables cuando corresponda (usando el módulo `typing`). Usa **mypy** para la comprobación estática de tipos. Todas las funciones deben pasar mypy sin errores.
- Incluye **docstrings** en funciones y clases siguiendo el PEP 257 (p. ej. estilo Google o NumPy) para documentar su propósito, parámetros y valores de retorno.

### V.2 Makefile

Incluye un `Makefile` en tu proyecto para automatizar tareas comunes. Debe contener las siguientes reglas (la regla `lint` obligatoria implica los flags indicados; se recomienda encarecidamente probar `--strict` para una comprobación más estricta):

- `install`: instalar las dependencias del proyecto usando pip, uv, pipx o cualquier otro gestor de paquetes que elijas.
- `run`: ejecutar el script principal del proyecto (p. ej. mediante el intérprete de Python).
- `debug`: ejecutar el script principal en modo depuración usando el depurador integrado de Python (p. ej. `pdb`).
- `clean`: eliminar archivos temporales o cachés (p. ej. `__pycache__`, `.mypy_cache`) para mantener limpio el entorno.
- `lint`: ejecutar los comandos `flake8 .` y `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`
- `lint-strict` (opcional): ejecutar los comandos `flake8 .` y `mypy . --strict`

### V.3 Directrices adicionales

- Crea programas de prueba para verificar la funcionalidad del proyecto (no se entregan ni se evalúan). Usa frameworks como `pytest` o `unittest` para tests unitarios, cubriendo casos límite.
- Incluye un archivo `.gitignore` para excluir los artefactos de Python.
- Se recomienda usar entornos virtuales (p. ej. `venv` o `conda`) para aislar las dependencias durante el desarrollo.

Si se aplican requisitos adicionales específicos del proyecto, se indicarán justo debajo de esta sección.

### V.4 Requisitos adicionales

Además de las reglas comunes de Python anteriores, este proyecto requiere:

- Tus modelos de datos deben usar **pydantic** para validación y seguridad de tipos: las estructuras que intercambias entre etapas, como las de la sección *Modelos de datos*. Las clases de servicio u orquestación (indexer, retriever, pipeline, etc.) no tienen por qué ser pydantic.
- El estándar flake8 también se aplica a los archivos del bonus.
- Debes usar **uv** como gestor de proyecto y de paquetes: el revisor y la moulinette solo ejecutan `uv sync`.
- Tu sistema debe exponer una **interfaz de línea de comandos (CLI)** construida con **Python Fire**, y barras de progreso (**tqdm**) para las operaciones largas.
- Debes usar el siguiente modelo:
  - `Qwen/Qwen3-0.6B` (por defecto)
  - Puedes usar otros modelos siempre que todo siga funcionando con `Qwen/Qwen3-0.6B`.
- Aparte de las herramientas exigidas arriba, puedes usar la librería que quieras.

---

## Capítulo VI — Parte obligatoria

Vas a construir un sistema de **Generación Aumentada por Recuperación** que responda preguntas sobre un código fuente. Ingieres el repositorio vLLM proporcionado en un índice consultable, recuperas los fragmentos más relevantes para una pregunta, generas una respuesta a partir de ellos con `Qwen/Qwen3-0.6B` y mides la calidad de la recuperación con recall@k. Tu sistema se juzga según si recupera las ubicaciones de origen correctas y produce respuestas fundamentadas en ellas.

### VI.1 Indexar el código fuente

Todo empieza por el índice. Lee los archivos que consideres útiles del repositorio vLLM incluido en los adjuntos, divide cada uno en fragmentos (*chunks*) y guarda de forma persistente un índice que la recuperación pueda consultar en milisegundos. **Indexar todo el corpus debe tardar como máximo 5 minutos.**

Un archivo Python y una página Markdown no se dividen de la misma manera, así que tu programa debe implementar **dos estrategias de fragmentación distintas**:

- Fragmentación de código Python,
- Fragmentación de Markdown / texto.

Para la recuperación en sí, implementa al menos uno de los dos métodos léxicos clásicos. La elección es tuya:

- **TF-IDF**,
- **BM25**.

Puedes explorar otros métodos adicionales, siempre que uno de estos dos esté implementado.

> El tamaño de los fragmentos se configura mediante un argumento de la CLI (`--max_chunk_size`), con un valor por defecto de **2000 caracteres**. No lo superes: la moulinette rechaza cualquier fuente recuperada de más de 2000 caracteres (su `max_context_length`), y una sola fuente demasiado larga invalida toda tu salida. Fragmentos más pequeños están bien; documenta el efecto que tienen sobre tu recall@k.

*(Figura: Indexar un corpus objetivo: el indexador fragmenta cada archivo y guarda el índice en `data/processed/`.)*

> Las preguntas y el código fuente rara vez usan las mismas palabras: una pregunta puede parafrasear una idea o citar un identificador literalmente. Lo que conserves en el momento de indexar decide cuál de las dos cosas podrás seguir encontrando.

### VI.2 Recuperación (retrieval)

Con el índice construido, ya puedes buscar en él. Dada una pregunta, tu sistema devuelve los **top-k** fragmentos más relevantes. Cada resultado es una ubicación de origen: un `file_path` y el rango de caracteres (`first_character_index`, `last_character_index`) que abarca, con un ancho máximo de 2000 caracteres.

> `file_path` debe coincidir exactamente con la ruta del corpus (p. ej. `data/raw/vllm-0.10.1/docs/features/lora.md`). El evaluador compara las rutas literalmente, así que una ruta con un prefijo distinto nunca coincide.

La recuperación debe funcionar tanto para una sola consulta como en lote sobre un dataset completo de preguntas leído desde JSON. En los datasets de referencia, tu sistema debe alcanzar **al menos un 80 % de recall@5 en las preguntas de documentación (docs) y un 50 % en las preguntas de código (code)** (la métrica se define en el capítulo de Evaluación).

*(Figura: Una búsqueda de una sola consulta devuelve ubicaciones de origen ordenadas, cada una con su ruta de archivo y su rango de caracteres.)*

### VI.3 Generación de respuestas

Con los fragmentos correctos recuperados, el sistema genera una respuesta en lenguaje natural usando `Qwen/Qwen3-0.6B`. Pasa el contexto recuperado al modelo respetando su presupuesto de tokens, y produce JSON estructurado siguiendo los modelos pydantic proporcionados.

Una respuesta satisfactoria es:

- Coherente y comprensible,
- Fundamentada en las fuentes recuperadas, sin alucinaciones importantes,
- Pertinente: responde a la pregunta que realmente se ha hecho.

> El modelo obligatorio `Qwen/Qwen3-0.6B` tiene limitaciones de razonamiento conocidas. Estos criterios describen el objetivo al que debe aspirar una respuesta, no un listón estricto de aprobado/suspenso: la evaluación prioriza la calidad de la recuperación, la fundamentación y la estrategia de prompt por encima de una redacción final perfecta. Una respuesta válida es coherente y está mayormente fundamentada en los documentos recuperados, aunque sea parcialmente incompleta por las limitaciones del modelo base.

### VI.4 Modelos de datos

Implementa los siguientes modelos pydantic; validan los datos que se intercambian entre las etapas. El modelo `MinimalSource` representa una única fuente de información:

**Modelo MinimalSource**
```python
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
```

Los modelos `UnansweredQuestion` y `AnsweredQuestion` representan una pregunta sin responder y una pregunta respondida:

**Modelos UnansweredQuestion y AnsweredQuestion**
```python
class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str

class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str
```

El modelo `RagDataset` representa un dataset de preguntas RAG:

**Modelo RagDataset**
```python
class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
```

Los modelos `MinimalSearchResults` y `MinimalAnswer` representan los resultados de búsqueda y una respuesta:

**Modelos MinimalSearchResults y MinimalAnswer**
```python
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    answer: str
```

Los modelos `StudentSearchResults` y `StudentSearchResultsAndAnswer` representan resultados de búsqueda y resultados de búsqueda con respuestas:

**Modelos StudentSearchResults y StudentSearchResultsAndAnswer**
```python
class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
```

Los modelos proporcionados son una base. Puedes ampliarlos añadiendo nuevos modelos o campos extra (por ejemplo, en el modelo de resultados de búsqueda) si tu implementación lo requiere.

### VI.5 Salida

Cada comando escribe un archivo JSON que se ajusta a los modelos pydantic proporcionados:

- **Para operaciones de búsqueda:** usa el modelo `StudentSearchResults` con:
  - `search_results`: lista de `MinimalSearchResults` que contiene `question_id`, `question` y `retrieved_sources`
  - `k`: número de resultados solicitados
- **Para la generación de respuestas:** usa el modelo `StudentSearchResultsAndAnswer` con:
  - `search_results`: lista de `MinimalAnswer` que contiene `question_id`, `question`, `retrieved_sources` y `answer`
  - `k`: número de resultados solicitados
- **Información de la fuente:** cada `MinimalSource` contiene:
  - `file_path`: ruta al archivo de origen, relativa a la raíz de tu proyecto y escrita exactamente igual que en el corpus ingerido (p. ej. `data/raw/vllm-0.10.1/...`); se compara literalmente con la referencia
  - `first_character_index`: posición del carácter inicial
  - `last_character_index`: posición del carácter final

#### Formato de salida

La salida debe respetar la base mínima de los modelos proporcionados, pero puede enriquecerse de la siguiente forma:

**Ejemplo: salida StudentSearchResults**
```json
"search_results": [
    {
        "question_id": "q1",
        "question": "How to configure OpenAI server?",
        "retrieved_sources": [
            {
               "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
               "first_character_index": 9867,
               "last_character_index": 10100
            },
            {
               "file_path": "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
               "first_character_index": 267,
               "last_character_index": 400
            }
        ]
    }
],
"k": 10
```

Para las respuestas, la salida debe seguir el modelo `StudentSearchResultsAndAnswer`:

**Ejemplo: salida StudentSearchResultsAndAnswer**
```json
"search_results": [
    {
        "question_id": "q1",
        "question": "How to configure OpenAI server?",
        "retrieved_sources": [
            {
               "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
               "first_character_index": 9867,
               "last_character_index": 10100
            },
            {
               "file_path": "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
               "first_character_index": 267,
               "last_character_index": 400
            }
        ],
        "answer": "To configure the OpenAI compatible server in vLLM..."
    }
],
"k": 10
```

### VI.6 Interfaz de línea de comandos

Proporciona una CLI construida con Python Fire. Cada comando se invoca como `uv run python -m src <command> [options]`. Los siguientes comandos son obligatorios (las opciones mostradas son el mínimo; puedes añadir las tuyas):

- `index --max_chunk_size <int>`
  Ingiere `data/raw/` y construye el índice en `data/processed/`.
- `search <query> --k <int>`
  Devuelve las top-k fuentes para una única consulta.
- `search_dataset --dataset_path <path> --k <int> --save_directory <dir>`
  Ejecuta la búsqueda sobre un dataset completo y escribe un archivo JSON `StudentSearchResults`.
- `answer <query> --k <int>`
  Responde a una única consulta usando el contexto recuperado.
- `answer_dataset --student_search_results_path <path> --save_directory <dir>`
  Genera respuestas para un dataset, produciendo un archivo JSON `StudentSearchResultsAndAnswer`.
- `evaluate --student_search_results_path <path> --dataset_path <path>`
  Informa de tu propio recall@k frente a un dataset de referencia (ground truth), para tus propias pruebas.

**Todas las rutas de entrada y salida deben ser argumentos configurables de la CLI y nunca estar escritas a fuego en el código.** Incluye barras de progreso (tqdm) para las operaciones largas, y gestiona con elegancia las entradas degeneradas (consulta vacía, consulta sin sentido, `k=0`, archivos inexistentes, JSON mal formado). La CLI se prueba con esos casos límite y **nunca debe cerrarse con un traceback no gestionado.**

> El comando `evaluate` es para tus propias iteraciones. El recall@k oficial usado durante la defensa lo calcula el ejecutable **moulinette** proporcionado, no tu código. Tu solución **nunca** debe importar ni llamar a la moulinette.

### VI.7 Recorrido completo de principio a fin

Esta sección muestra el flujo de trabajo completo, desde los documentos en bruto hasta la evaluación y la generación de respuestas, exactamente como lo ejecutan los scripts de examen de referencia.

#### VI.7.1 Estructura de proyecto esperada para la evaluación

Durante la defensa, el pipeline de recuperación se ejecuta de principio a fin como un único flujo automatizado (`index` → `search_dataset` → moulinette `evaluate_student_search_results`) usando scripts de examen de referencia. Los comandos individuales siguen disponibles para depurar, pero para que la ejecución automatizada sea reproducible sin reestructurar nada a mano, tu repositorio debe respetar la siguiente estructura y convenciones:

- `src/`: tu módulo Python, ejecutable como `uv run python -m src <command>`.
- `pyproject.toml` y `uv.lock` en la raíz del repositorio, para que `uv sync` funcione desde la raíz.
- `README.md` en la raíz del repositorio.
- `data/raw/`: las fuentes indexadas (el repositorio vLLM proporcionado).
- `data/processed/`: el índice generado por el comando `index`.
- `data/datasets/UnansweredQuestions/` y `data/datasets/AnsweredQuestions/`: los datasets de preguntas.
- `data/output/search_results/<DatasetScope>/`: la salida de `search_dataset`, separada por dataset.
- `data/output/search_results_and_answer/<DatasetScope>/`: la salida de `answer_dataset`, separada por dataset.

Todas las rutas de entrada y salida (`--dataset_path`, `--save_directory`, `--student_search_results_path`) deben ser argumentos configurables de la CLI y nunca deben estar escritas a fuego: el evaluador las apunta a los datasets y a directorios de salida dedicados. Si no se respeta la estructura o la interfaz de comandos, los scripts de referencia no podrán ejecutarse y las comprobaciones correspondientes se considerarán fallidas por diseño.

#### VI.7.2 Ejecutar el pipeline completo

El pipeline se maneja con cuatro comandos, en orden: indexar el corpus, buscar sobre un dataset completo, puntuar los resultados con la moulinette y, por último, generar respuestas. Los comandos de consulta única `search` y `answer` mostrados antes se comportan igual, pero con una pregunta cada vez.

**1. Indexar el corpus una vez:**
```
uv run python -m src index --max_chunk_size 2000
Ingestion complete! Indices saved under data/processed/
```

**2. Buscar sobre un dataset.** Separa siempre `--save_directory` por dataset (`UnansweredQuestions` o `AnsweredQuestions`): los datasets públicos comparten nombres de archivo, así que escribir todas las ejecuciones en la misma carpeta sobrescribiría los resultados anteriores.
```
uv run python -m src search_dataset
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
  --k 10
  --save_directory data/output/search_results/UnansweredQuestions
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json
```

**3. Puntuar con la moulinette** (renombra antes `moulinette-ubuntu`/`-fedora` a `moulinette`). Primero van los resultados del estudiante y en segundo lugar el dataset de referencia `AnsweredQuestions`:
```
./moulinette evaluate_student_search_results
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json
  data/datasets/AnsweredQuestions/dataset_docs_public.json
  --k 10 --max_context_length 2000
Student data is valid: True
Evaluation Results
========================================
Recall@1: 0.450 Recall@3: 0.590 Recall@5: 0.650 Recall@10: 0.720
```

> Estas cifras solo ilustran el formato de salida. No son una puntuación de referencia. Los umbrales que debes alcanzar están definidos en el capítulo de Evaluación.

**4. Generar respuestas a partir de los resultados de búsqueda:**
```
uv run python -m src answer_dataset
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
Loaded 100 questions ... Processed 100 of 100 questions
Saved student_search_results_and_answer to .../UnansweredQuestions/dataset_docs_public.json
```

---

## Capítulo VII — Evaluación

### VII.1 Métricas de evaluación

La calidad de la recuperación se mide con una métrica **recall@k**.

#### VII.1.1 Cálculo de Recall@k

Para cada pregunta, recall@k es la proporción de sus fuentes correctas que recuperas dentro de tus k primeros resultados. Una fuente correcta cuenta como encontrada cuando uno de tus resultados está **en el mismo archivo** y **se solapa con su rango de caracteres**.

El listón de solapamiento es bajo (un **IoU de 0,05**), así que no necesitas coincidir exactamente con el rango de referencia: basta con recuperar un fragmento que cubra la zona correcta del archivo correcto. Un resultado en un archivo distinto nunca cuenta, por eso `file_path` debe ser exacto.

*(Figura: La moulinette valida tu salida y después informa del recall en varios valores de k. Docs debe alcanzar un 80 % de recall@5; code, un 50 %.)*

#### VII.1.2 Rendimiento

Tu sistema debe respetar unos rendimientos mínimos, enumerados a continuación:

- **Tiempo de indexación:** como máximo 5 minutos para todo el corpus.
- **Rendimiento de recuperación:** como máximo 90 segundos para 200 preguntas.
- **Recall@5:** al menos un 80 % en preguntas de docs y un 50 % en preguntas de código.

---

## Capítulo VIII — Requisitos del README

Debe incluirse un archivo `README.md` en la raíz de tu repositorio Git. Su propósito es permitir que cualquiera que no conozca el proyecto (compañeros, staff, reclutadores, etc.) entienda rápidamente de qué trata, cómo ejecutarlo y dónde encontrar más información sobre el tema.

El `README.md` debe incluir como mínimo:

- La **primera línea** debe estar en cursiva y decir: *This project has been created as part of the 42 curriculum by <login1>[, <login2>[, <login3>[...]]].*
- Una sección **"Description"** que presente claramente el proyecto, incluyendo su objetivo y una breve visión general.
- Una sección **"Instructions"** con toda la información relevante sobre compilación, instalación y/o ejecución.
- Una sección **"Resources"** que enumere referencias clásicas relacionadas con el tema (documentación, artículos, tutoriales, etc.), así como una descripción de cómo se ha usado la IA, especificando para qué tareas y en qué partes del proyecto.

➠ Puede que se exijan secciones adicionales según el proyecto (p. ej. ejemplos de uso, lista de funcionalidades, decisiones técnicas, etc.).

Cualquier añadido obligatorio se enumerará explícitamente a continuación.

Para este proyecto, el `README.md` también debe incluir:

- **System architecture (arquitectura del sistema):** describe los componentes de tu pipeline RAG y cómo interactúan.
- **Chunking strategy (estrategia de fragmentación):** explica tu enfoque para segmentar los documentos.
- **Retrieval method (método de recuperación):** detalla el algoritmo de recuperación y el mecanismo de ordenación.
- **Performance analysis (análisis de rendimiento):** comenta las puntuaciones de recall@k y el rendimiento del sistema.
- **Design decisions (decisiones de diseño):** explica las decisiones clave de implementación.
- **Challenges faced (dificultades encontradas):** documenta las dificultades encontradas y sus soluciones.
- **Example usage (ejemplos de uso):** proporciona ejemplos claros de cómo ejecutar tu sistema.

> **Tu README debe estar escrito en inglés.**

---

## Capítulo IX — Parte bonus

> El bonus solo se evalúa una vez validada toda la parte obligatoria: todos los requisitos obligatorios cumplidos y el sistema robusto incluso cuando un revisor lo fuerza en direcciones raras. Hasta que la parte obligatoria no esté validada por completo, el bonus no se evalúa en absoluto.

Hay cinco bonus, cada uno vale un punto. Todos se ejecutan en una máquina del campus solo con CPU, y cada uno amplía el sistema obligatorio en una dirección con la que te volverás a encontrar más adelante. Un bonus solo cuenta si está implementado y funciona, no si simplemente se describe en el `README.md`, y puede que se te pida hacer una demostración.

1. **Embeddings semánticos:** añade un índice vectorial construido con un modelo ligero de CPU (como `all-MiniLM-L6-v2`) junto a tu índice léxico.
2. **Recuperación híbrida:** combina las clasificaciones léxica y semántica en una única lista de resultados.
3. **Indexación incremental:** cuando un archivo cambie, reindexa solo ese archivo en lugar de reconstruir todo el índice.
4. **Caché:** cachea el índice y los resultados de las consultas para acelerar el arranque en frío y las consultas repetidas.
5. **API HTTP local:** expón la consulta del índice y la respuesta a preguntas a través de una pequeña API HTTP local, para que el sistema se pueda manejar desde algo que no sea la CLI.

---

## Capítulo X — Entrega y evaluación entre pares

Entrega tu trabajo en tu repositorio Git como de costumbre. Solo se evaluará el trabajo que esté dentro de tu repositorio durante la defensa. No dudes en comprobar dos veces los nombres de tus archivos para asegurarte de que son correctos.

Tu repositorio debe contener:

- El directorio `src/` con tu implementación
- `pyproject.toml` y `uv.lock` para la gestión de dependencias
- Un `Makefile` que exponga las reglas de las Instrucciones comunes (`install`, `run`, `debug`, `clean`, `lint`)
- `README.md` con documentación completa
- Cualquier archivo de configuración adicional necesario para ejecutar tu solución

> **No incluyas archivos de datos grandes, pesos de modelos ni salidas generadas en tu repositorio.** El evaluador los generará durante el proceso de evaluación. El stack de deep learning y los pesos del modelo pueden sumar varios gigabytes, así que construye y ejecuta tu proyecto desde una ubicación con suficiente espacio libre en disco.

### X.1 Instrucciones de recodificación

Durante la evaluación, puede que ocasionalmente se pida una breve modificación del proyecto. Puede tratarse de un pequeño cambio de comportamiento, unas pocas líneas de código que escribir o reescribir, o una funcionalidad fácil de añadir.

Aunque este paso puede no aplicarse a todos los proyectos, debes estar preparado para él si se menciona en las directrices de evaluación.

Este paso sirve para verificar tu comprensión real de una parte concreta del proyecto. La modificación puede hacerse en el entorno de desarrollo que elijas (p. ej. tu configuración habitual) y debería poder hacerse en unos pocos minutos, salvo que se defina un plazo concreto como parte de la evaluación.

Por ejemplo, se te puede pedir que hagas una pequeña actualización de una función o script, que modifiques una visualización, o que ajustes una estructura de datos para almacenar información nueva, etc.

Los detalles (alcance, objetivo, etc.) se especificarán en las directrices de evaluación y pueden variar de una evaluación a otra para el mismo proyecto.
