# Explicación detallada: qué tienes que hacer exactamente

> Este documento no contiene código ni pseudocódigo. Su objetivo es que entiendas cada pieza del proyecto, por qué existe y cómo encaja con las demás, para que después seas tú quien decida cómo implementarla.

---

## 1. El problema que resuelves

Imagina que alguien te pregunta: *"¿Cómo activo los adaptadores LoRA en vLLM?"*. Tú no te sabes vLLM de memoria, pero tienes su código y documentación en una carpeta. Lo que harías es:

1. Buscar en la carpeta los archivos que hablan de LoRA.
2. Abrir los dos o tres párrafos más relevantes.
3. Leerlos y responder con tus palabras basándote en ellos.

Tu programa tiene que hacer exactamente eso, de forma automática y para cientos de preguntas. El paso 1 y 2 es la **recuperación** (la parte importante y evaluada con números). El paso 3 es la **generación** (la hace un modelo de lenguaje pequeño, Qwen3-0.6B).

¿Por qué no preguntarle directamente al modelo? Porque un modelo de 0,6 mil millones de parámetros sabe muy poco de vLLM y se inventaría la respuesta. Si le das los párrafos correctos delante, solo tiene que "leer y resumir", algo que sí sabe hacer razonablemente.

---

## 2. Visión global del sistema

Tu proyecto es una herramienta de línea de comandos con seis órdenes. Por dentro, tiene estas piezas conceptuales:

1. **Lector de archivos**: recorre la carpeta del corpus y lee los archivos que decidas incluir.
2. **Troceador (chunker)**: parte cada archivo en fragmentos de como mucho N caracteres, de forma distinta según sea Python o Markdown.
3. **Indexador**: convierte los fragmentos en una estructura que permita buscar rápido (TF-IDF o BM25) y la guarda en disco.
4. **Buscador (retriever)**: carga el índice, recibe una pregunta y devuelve los k fragmentos más parecidos, expresados como "archivo + posición inicial + posición final".
5. **Generador**: carga el modelo Qwen, construye un prompt con la pregunta y los fragmentos, y obtiene una respuesta.
6. **Evaluador**: compara tus resultados con las respuestas correctas y calcula el recall@k (solo para ti; la nota oficial la calcula la moulinette).
7. **Modelos de datos**: las clases pydantic que definen la forma exacta de los JSON que entran y salen.
8. **CLI**: la capa que conecta todo lo anterior con los comandos que escribe el usuario.

Cada pieza alimenta a la siguiente, y por eso el subject insiste en leerlo entero antes de empezar.

---

## 3. Preparación del entorno

### 3.1 Estructura de carpetas

El evaluador ejecuta scripts automáticos que asumen una estructura concreta. Si no la respetas, fallan por diseño. Necesitas en la raíz del repositorio:

- Una carpeta `src` que sea un módulo Python ejecutable con `python -m src`. Eso implica que dentro debe existir un punto de entrada que Python ejecuta cuando se llama al módulo así.
- Los archivos de configuración de `uv` (`pyproject.toml` y `uv.lock`).
- Un `Makefile`, un `README.md` y un `.gitignore`.
- Una carpeta `data` con subcarpetas: `raw` (el corpus, es decir vLLM), `processed` (tu índice), `datasets` (las preguntas) y `output` (tus resultados).

La carpeta `vllm-0.10.1` debe quedar en `data/raw/vllm-0.10.1`. Esto es crítico, porque la ruta que devuelves en cada resultado se compara letra a letra con la ruta correcta.

### 3.2 uv

`uv` es un gestor de proyectos y paquetes de Python, mucho más rápido que pip. El evaluador solo va a ejecutar `uv sync`, que lee tu `pyproject.toml` y tu `uv.lock` e instala exactamente las mismas versiones que tú. Por tanto, todas las dependencias (pydantic, fire, tqdm, la librería para cargar el modelo, la de BM25 si usas una, etc.) deben estar declaradas ahí, y el archivo de bloqueo debe estar subido al repositorio. Después, todo se ejecuta con `uv run`, que usa ese entorno.

Ten en cuenta que las librerías de deep learning (como PyTorch) y el modelo pesan varios gigas. Trabaja en un disco con espacio y **no subas** ni el modelo ni el entorno virtual al repositorio.

### 3.3 Makefile

Es un atajo para tareas repetitivas. Te exigen cinco reglas: instalar dependencias, ejecutar el programa, ejecutarlo con el depurador de Python, limpiar cachés y pasar los linters (flake8 y mypy con unos flags concretos que aparecen en el subject). Hay una sexta opcional con mypy en modo estricto.

### 3.4 Calidad de código

Todo el código debe pasar flake8 (estilo) y mypy (tipos). Esto significa que cada función necesita anotaciones de tipo en parámetros y retorno, y docstrings explicando qué hace. Conviene configurar esto desde el primer día; arreglarlo al final es muy tedioso.

### 3.5 El .gitignore

Debe excluir los artefactos de Python (cachés, entornos virtuales) y además las carpetas de datos pesados: el corpus, el índice generado, las salidas y el modelo descargado.

---

## 4. Los modelos de datos (pydantic)

Pydantic es una librería que permite definir "formas" de datos: qué campos tiene un objeto y de qué tipo es cada uno. Cuando creas un objeto o lees un JSON, pydantic comprueba que cumple esa forma y, si no, lanza un error claro. Esto te protege de JSON mal formados y te permite generar la salida con el formato exacto que espera la moulinette.

Tienes que definir estos modelos (sus nombres y campos deben ser los del subject):

- **MinimalSource**: una fuente. Tres datos: la ruta del archivo, la posición del primer carácter y la del último. Es la "respuesta" de la búsqueda: *"lo que buscas está en tal archivo, entre el carácter 9867 y el 10100"*.
- **UnansweredQuestion**: una pregunta con su identificador. Si el identificador no viene, se genera uno aleatorio automáticamente.
- **AnsweredQuestion**: lo mismo, más la lista de fuentes correctas y la respuesta correcta. Es el formato del dataset de referencia (la "hoja de soluciones").
- **RagDataset**: un contenedor con una lista de preguntas (respondidas o no). Es la forma de los archivos de datasets que te dan.
- **MinimalSearchResults**: el resultado de buscar una pregunta: su id, su texto y la lista de fuentes que tú has recuperado.
- **MinimalAnswer**: lo mismo, más la respuesta generada por el modelo.
- **StudentSearchResults**: el archivo final de `search_dataset`: una lista de MinimalSearchResults y el valor de k usado.
- **StudentSearchResultsAndAnswer**: el archivo final de `answer_dataset`: una lista de MinimalAnswer y el valor de k.

Puedes añadir campos o modelos extra (por ejemplo, la puntuación de cada resultado o el texto del fragmento), pero sin quitar ni renombrar los obligatorios. Las clases que "hacen cosas" (indexador, buscador, pipeline) no tienen por qué ser pydantic.

---

## 5. Fase 1: indexación (comando `index`)

### 5.1 Qué archivos leer

Recorres recursivamente `data/raw`. Tú decides qué archivos merecen la pena. Como las preguntas son de dos tipos (documentación y código), lo lógico es incluir como mínimo los `.py` y los `.md`. Puedes probar a añadir otros tipos de texto y medir si el recall sube o baja. Imágenes, binarios o C++ probablemente solo añaden ruido.

Lee los archivos con cuidado: alguno puede tener una codificación rara o no poder abrirse. Tu programa no debe romperse por eso; simplemente se salta ese archivo.

**Muy importante**: la ruta que guardes para cada archivo debe ser relativa a la raíz del proyecto y empezar por `data/raw/vllm-0.10.1/...`, exactamente así. Una ruta absoluta (`/home/...`) o con otro prefijo nunca coincidirá.

### 5.2 Qué es un chunk y por qué hace falta

Un archivo entero puede tener decenas de miles de caracteres. No puedes devolverlo como resultado (la moulinette rechaza cualquier fuente de más de 2000 caracteres) ni meterlo entero en el modelo (no le cabe y se perdería). Por eso lo partes en fragmentos. Cada fragmento debe recordar:

- De qué archivo viene.
- En qué carácter empieza y en cuál termina dentro de ese archivo (posiciones en el texto original, contando desde cero).
- Su texto.

Esas posiciones son las que luego devolverás como `first_character_index` y `last_character_index`. Tienen que referirse al archivo original sin modificar, por eso nunca debes alterar el corpus, y si normalizas el texto (quitar espacios, etc.) debe ser solo para buscar, no para calcular posiciones.

### 5.3 El tamaño máximo

El tamaño máximo lo decide el usuario con `--max_chunk_size` y por defecto es 2000. Ningún fragmento debe superarlo nunca: con uno solo que se pase, toda tu salida se considera inválida. Asegúrate de ello con una comprobación final, aunque tu estrategia "no debería" generar fragmentos grandes.

Fragmentos más pequeños también son válidos. Hay un equilibrio:

- **Fragmentos grandes**: más contexto por resultado y más probabilidad de cubrir la zona correcta, pero la búsqueda es menos precisa (mucho texto diluye las palabras clave) y llenan antes el contexto del modelo.
- **Fragmentos pequeños**: búsqueda más precisa, pero pueden cortar una idea por la mitad, y hay más fragmentos que se "pelean" por los k puestos.

El subject te pide que en el README comentes el efecto del tamaño sobre tu recall, así que prueba varios valores y apunta resultados.

### 5.4 Estrategia de troceado para Markdown

La documentación tiene una estructura natural: títulos, secciones, subsecciones, párrafos, bloques de código, listas. La idea es cortar respetando esa estructura:

- Lo ideal es que cada fragmento sea una sección (desde un título hasta el siguiente).
- Si una sección es demasiado grande, se subdivide por párrafos o líneas en blanco.
- Si incluso un párrafo es demasiado grande, se corta por líneas o, como último recurso, por número de caracteres.
- Si hay secciones muy pequeñas seguidas, puedes juntarlas para no tener fragmentos minúsculos sin contexto.

Una mejora habitual es que cada fragmento "sepa" bajo qué título está (por ejemplo, guardar el título del documento y el de la sección como información extra para la búsqueda), porque las preguntas suelen usar palabras que aparecen en los títulos.

También se usa a veces el **solapamiento** (overlap): que cada fragmento repita un poco del final del anterior, para no perder ideas que caen justo en el corte.

### 5.5 Estrategia de troceado para Python

El código tiene otra estructura: importaciones, clases, funciones, métodos, docstrings, comentarios. Cortar a mitad de una función es mala idea. La estrategia natural es:

- Cada función o clase de nivel superior es una unidad candidata a fragmento.
- Si una clase es muy grande, se divide por sus métodos.
- Si una función es enorme, se subdivide por bloques de líneas.
- El código "suelto" (importaciones, constantes, código entre funciones) forma sus propios fragmentos o se agrupa.

Python trae en su biblioteca estándar herramientas para analizar la estructura de un archivo de código y saber en qué línea empieza y acaba cada función o clase. A partir de las líneas puedes calcular las posiciones en caracteres. Ten un plan B para archivos que no se puedan analizar (con errores de sintaxis, por ejemplo): trocearlos de forma genérica por líneas.

Igual que en Markdown, ayuda mucho que el fragmento lleve información extra para la búsqueda: nombre de la clase y la función, nombre y ruta del archivo. Muchas preguntas de código mencionan identificadores concretos, y la ruta del archivo (por ejemplo, que contenga "lora" o "openai") es una pista muy valiosa.

### 5.6 El índice léxico: TF-IDF o BM25

Tienes que implementar al menos uno. Ambos son métodos **léxicos**: puntúan un fragmento según las palabras que comparte con la pregunta.

- **Tokenización**: primero conviertes cada texto en una lista de "términos". Aquí hay decisiones importantes: pasar a minúsculas, qué hacer con la puntuación, si quitar palabras muy comunes ("the", "is"), si reducir palabras a su raíz ("running" → "run"). En código hay una particularidad: los identificadores como `max_model_len` o `LoRARequest` conviene partirlos en sus piezas ("max", "model", "len"; "lora", "request") **y además** conservar el identificador completo. Así casas tanto preguntas que parafrasean como preguntas que citan el nombre exacto. A esto se refiere el subject con "lo que conserves al indexar decide qué podrás encontrar".
- **TF (frecuencia del término)**: cuántas veces aparece una palabra en un fragmento. Más apariciones → más relevante.
- **IDF (frecuencia inversa en documentos)**: cuán rara es la palabra en todo el corpus. Palabras rarísimas ("speculative") valen mucho; palabras que salen en todas partes ("self", "the") valen casi nada.
- **TF-IDF**: combina ambas y representa cada fragmento y la pregunta como vectores; la relevancia es lo parecidos que son (normalmente, similitud del coseno).
- **BM25**: una evolución de TF-IDF que suele funcionar mejor. Añade dos ideas: la saturación (que una palabra aparezca 50 veces no es 50 veces mejor que 1) y la normalización por longitud (un fragmento largo no gana solo por ser largo). Tiene dos parámetros ajustables, llamados habitualmente k1 y b.

Puedes implementarlo tú o usar una librería; el subject permite cualquier librería. Lo importante es que entiendas cómo funciona, porque en la defensa te lo preguntarán y pueden pedirte modificaciones.

### 5.7 Persistir el índice

Construir el índice tarda; buscar debe ser rapidísimo. Por eso `index` guarda en `data/processed` todo lo necesario (los fragmentos con su ruta, posiciones y texto, y la estructura del índice) y los comandos de búsqueda solo cargan eso, sin volver a leer el corpus. El formato de guardado es libre.

### 5.8 Requisitos de esta fase

- Todo el corpus en menos de 5 minutos.
- Barra de progreso (tqdm) mientras se procesan archivos.
- Mensaje final indicando que ha terminado y dónde se ha guardado.
- Si `data/raw` no existe o está vacío, mensaje de error claro, sin traceback.
- Si el tamaño pedido es cero, negativo o absurdo, error claro.

---

## 6. Fase 2: recuperación (comandos `search` y `search_dataset`)

### 6.1 Búsqueda de una sola pregunta

`search` recibe una pregunta y un k. Carga el índice, tokeniza la pregunta con **exactamente las mismas reglas** que usaste al indexar (si no, no coincidirán los términos), puntúa los fragmentos, los ordena de mayor a menor y devuelve los k primeros como fuentes: ruta, carácter inicial y carácter final. La forma de mostrar el resultado por pantalla es libre, pero debe ser clara.

### 6.2 Búsqueda sobre un dataset

`search_dataset` recibe:

- La ruta de un JSON con preguntas (formato RagDataset).
- El valor de k.
- La carpeta donde guardar el resultado.

Lee y valida el JSON, busca cada pregunta (con barra de progreso), y escribe un archivo StudentSearchResults en la carpeta indicada, **con el mismo nombre de archivo que el dataset de entrada** (así lo muestra el ejemplo del subject). Por eso se usan carpetas distintas para UnansweredQuestions y AnsweredQuestions: los datasets se llaman igual y se pisarían.

Si la carpeta de salida no existe, créala. Nunca escribas rutas fijas en el código: todo viene de argumentos.

### 6.3 Rendimiento

200 preguntas en menos de 90 segundos. Esto significa cargar el índice **una sola vez** y reutilizarlo para todas las preguntas, no recargarlo en cada una.

### 6.4 Cómo mejorar el recall

Esta es la parte donde más tiempo vas a iterar. Ideas que puedes probar, midiendo siempre el efecto:

- Ajustar el tamaño de los fragmentos y el solapamiento.
- Mejorar la tokenización (partir identificadores, quitar palabras vacías, reducir a raíces).
- Añadir a cada fragmento su ruta de archivo y los títulos o nombres de función como texto buscable.
- Dar distinto peso a distintas partes (título vs. cuerpo).
- Ajustar los parámetros de BM25.
- Evitar que el top-k se llene de fragmentos casi idénticos del mismo archivo.
- Detectar si la pregunta parece de código o de documentación y favorecer un tipo de archivo u otro.

Estudia los fallos: coge preguntas donde no aciertas, mira cuál era la fuente correcta y pregúntate por qué tu fragmento no quedó arriba. Esa es la forma más eficaz de mejorar.

---

## 7. Fase 3: generación (comandos `answer` y `answer_dataset`)

### 7.1 El modelo y la librería para usarlo

Hay que distinguir tres cosas que se suelen confundir:

- **El modelo** es `Qwen/Qwen3-0.6B`: un archivo de pesos (algo más de 1 GB) publicado en Hugging Face. No es una librería ni un programa; solo son números. Es obligatorio.
- **La librería** es el código Python que sabe cargar esos pesos y ejecutarlos. El subject no impone ninguna ("puedes usar la librería que quieras"). La opción estándar es **`transformers`** de Hugging Face, que funciona sobre **PyTorch** (`torch`).
- **El tokenizador** viene con el modelo y convierte texto en números (tokens) y al revés. Lo carga la misma librería.

**Relación con call-me-maybe.** En ese proyecto 42 te dio un paquete llamado `llm_sdk` con una clase `Small_LLM_Model`. Por dentro, esa clase es solo un envoltorio de `transformers` y `torch` que carga Qwen3-0.6B y expone unos pocos métodos: codificar texto, decodificar tokens y obtener los *logits* del siguiente token. Tú construiste encima un bucle que genera token a token y decodificación restringida. En este proyecto **no se entrega ese paquete** y no es obligatorio usarlo. Además, no es buena idea reutilizarlo tal cual. Su método de logits vuelve a procesar el prompt entero en cada token nuevo, y aquí los prompts llevan miles de tokens de contexto. En CPU sería lentísimo, y tienes que generar cientos de respuestas.

**Qué usar entonces.** `transformers` directamente, con sus piezas de alto nivel:

- La clase automática de tokenizador y la de modelo causal (las mismas que usa `llm_sdk` por dentro), a las que pasas el nombre `Qwen/Qwen3-0.6B`.
- El método de la plantilla de chat del tokenizador, que construye el prompt con los roles (sistema, usuario, asistente). Así no tienes que escribir a mano los marcadores `<|im_start|>` como hacías en call-me-maybe. Admite una opción para desactivar el modo pensamiento de Qwen3.
- El método de generación del modelo, que hace el bucle token a token por ti, de forma eficiente (reutiliza cálculos previos) y con opciones de longitud máxima, temperatura, etc.

Lo que sí puedes aprovechar de call-me-maybe son las ideas: qué es un tokenizador, qué es la plantilla de chat, por qué hay que quitar el bloque de razonamiento y cómo detectar el final de la respuesta.

**¿Y vLLM?** La carpeta `vllm-0.10.1` es, casualmente, también una librería para ejecutar LLMs. Pero en este proyecto es solo el corpus sobre el que se hacen preguntas. No la instales como motor. Está pensada para GPUs y servidores, es muy pesada y no aporta nada para un modelo tan pequeño en CPU.

**Dependencias.** Añade `transformers` y `torch` a tu `pyproject.toml` con uv. Si tu máquina no tiene GPU NVIDIA, puedes configurar uv para que use la versión de PyTorch solo para CPU, que ocupa mucho menos que la de CUDA (lo explica la documentación de uv, en su guía de integración con PyTorch). La primera ejecución descarga el modelo en la caché de Hugging Face (por defecto en tu carpeta personal). En las máquinas de 42 con poco espacio conviene apuntar esa caché a `goinfre` mediante una variable de entorno, igual que hacías con el `.venv` en call-me-maybe.

Para entender PyTorch en sí (tensores, dispositivos, precisión, gradientes) y practicarlo, consulta [06_pytorch.md](06_pytorch.md).

**Una vez cargado, reutilízalo.** Cargar el modelo tarda unos segundos. Igual que el índice, cárgalo una sola vez por ejecución y reutilízalo para todas las preguntas.

Qwen3 tiene un "modo pensamiento" (thinking) en el que antes de responder escribe su razonamiento. Para este proyecto normalmente interesa desactivarlo o eliminar esa parte de la salida, porque consume tokens y ensucia la respuesta. Investiga cómo se controla en la documentación del modelo.

### 7.2 La plantilla de chat y el prompt

Los modelos de chat esperan los mensajes en un formato concreto (quién habla: sistema, usuario, asistente). El tokenizador trae una "plantilla de chat" que lo construye por ti. Tu prompt tendrá normalmente:

- Un mensaje de sistema con instrucciones: responde solo con la información del contexto, si no está di que no lo sabes, sé conciso, etc.
- Un mensaje de usuario con los fragmentos recuperados (idealmente indicando de qué archivo viene cada uno) y la pregunta.

La **estrategia de prompt** se evalúa explícitamente, así que merece la pena pensarla y probar variantes.

### 7.3 El presupuesto de tokens

El modelo tiene una ventana de contexto limitada, y cuanto más texto le metes, más lento va y más se "pierde" (sobre todo uno tan pequeño). "Respetar el presupuesto de tokens" significa: contar los tokens del prompt (con el tokenizador) y, si te pasas, meter menos fragmentos o recortarlos, empezando por los menos relevantes. También debes limitar cuántos tokens puede generar como respuesta.

### 7.4 Parámetros de generación

Al generar puedes controlar la aleatoriedad (temperatura), la longitud máxima y otros. Para respuestas fieles a las fuentes suele ir mejor poca aleatoriedad.

### 7.5 `answer`

Recibe una pregunta y un k: busca (fase 2), construye el prompt, genera y muestra la respuesta junto con las fuentes usadas.

### 7.6 `answer_dataset`

**No** recibe un dataset de preguntas, sino **el archivo de resultados de `search_dataset`** (StudentSearchResults). Para cada entrada ya tienes la pregunta y las fuentes; con las posiciones de cada fuente recortas el texto del archivo original, construyes el prompt y generas. El resultado se guarda como StudentSearchResultsAndAnswer, con el mismo nombre de archivo, en la carpeta indicada. Lleva barra de progreso, porque en CPU puede tardar bastante.

### 7.7 Qué es una buena respuesta

Coherente, basada en las fuentes (sin inventar) y que responda a lo preguntado. Con un modelo tan pequeño no se exige perfección: se valora más que la recuperación sea buena y el prompt esté bien pensado.

---

## 8. Fase 4: evaluación

### 8.1 Qué es recall@k

Para cada pregunta, el dataset de referencia dice cuáles son sus fuentes correctas (archivo + rango). Recall@k es: de esas fuentes correctas, ¿qué fracción aparece entre tus k primeros resultados? Luego se hace la media sobre todas las preguntas.

Ejemplo mental: una pregunta tiene 2 fuentes correctas; entre tus 5 primeros resultados aciertas una → recall@5 de esa pregunta = 0,5.

### 8.2 Qué significa "acertar"

Un resultado tuyo acierta una fuente correcta si:

1. Está en **exactamente el mismo archivo** (ruta idéntica), y
2. Su rango de caracteres se **solapa** lo suficiente con el correcto, medido con **IoU** (intersección sobre unión): la longitud de la parte común dividida entre la longitud total que cubren ambos rangos juntos. Basta con que sea al menos 0,05, un listón muy bajo.

Ojo con este detalle: como el IoU divide por la unión, un fragmento enorme que contiene una fuente correcta pequeña puede quedar por debajo de 0,05. Es otro motivo para no hacer fragmentos gigantes.

### 8.3 El comando `evaluate`

Es para ti: recibe tu archivo de resultados y el dataset con respuestas correctas, y calcula tu recall@k (conviene mostrarlo para varios k: 1, 3, 5, 10). Te permite iterar rápido sin la moulinette. Debería dar números parecidos a los de la moulinette; si no, probablemente estás calculando mal el solapamiento.

### 8.4 La moulinette

Es un ejecutable que te dan. Valida que tu JSON es correcto (rutas, tamaños ≤ 2000, formato) y calcula el recall oficial. Tu código **nunca** debe importarla ni llamarla; se usa a mano desde la terminal.

### 8.5 Objetivos

- Documentación: recall@5 ≥ 80 %.
- Código: recall@5 ≥ 50 %.

Las preguntas de código son más difíciles porque la gente pregunta en lenguaje natural y el código usa identificadores.

---

## 9. La CLI con Python Fire

Fire convierte automáticamente una clase o un conjunto de funciones en comandos de terminal: el nombre del método es el comando y sus parámetros son las opciones. Los seis comandos deben llamarse exactamente `index`, `search`, `search_dataset`, `answer`, `answer_dataset` y `evaluate`, con las opciones que indica el subject (puedes añadir más, con valores por defecto sensatos).

### Robustez: nunca un traceback

Durante la defensa intentarán romper tu programa. Tienes que prever y responder con un mensaje claro a:

- Pregunta vacía o solo con espacios.
- Pregunta sin sentido (letras aleatorias, emojis): debe devolver algo o decir que no encuentra nada, pero sin romper.
- k igual a cero, negativo, no numérico, o mayor que el número de fragmentos.
- Archivos o carpetas que no existen.
- JSON mal formado o que no cumple el modelo esperado.
- Buscar sin haber indexado antes.
- Fallos al cargar el modelo (por ejemplo, sin conexión la primera vez).
- Interrumpir con Ctrl+C.

Una buena práctica es tener un único punto de captura de errores alrededor de toda la ejecución que convierta cualquier error inesperado en un mensaje amable, además de las validaciones concretas.

---

## 10. El README

En inglés. Primera línea en cursiva con la frase exacta de 42 y tu login. Secciones obligatorias: Description, Instructions, Resources (incluyendo cómo has usado la IA), System architecture, Chunking strategy, Retrieval method, Performance analysis (tus números de recall, tiempos, efecto del tamaño de chunk), Design decisions, Challenges faced y Example usage.

Consejo: ve apuntando desde el principio las decisiones que tomas y los resultados de cada experimento. Al final tendrás el README medio escrito.

---

## 11. Orden de trabajo recomendado

1. Preparar el entorno: estructura de carpetas, uv, Makefile, .gitignore, linters funcionando.
2. Mover vLLM a `data/raw` y conseguir los datasets y la moulinette.
3. Escribir los modelos pydantic y comprobar que puedes leer los datasets con ellos.
4. Hacer la CLI vacía con los seis comandos (que de momento solo muestren un mensaje).
5. Implementar un troceado sencillo por caracteres y un índice BM25/TF-IDF. Conseguir que `search` y `search_dataset` funcionen de principio a fin.
6. Implementar `evaluate` y pasar la moulinette. Ya tienes una línea base de recall.
7. Implementar los dos troceados inteligentes (Python y Markdown) y medir la mejora.
8. Iterar sobre tokenización y metadatos hasta superar los objetivos.
9. Implementar la generación con Qwen: `answer` y luego `answer_dataset`.
10. Endurecer la robustez probando todos los casos límite.
11. Revisar tiempos (indexación y búsqueda).
12. Escribir el README.
13. Solo si todo lo anterior es perfecto: bonus.

---

## 12. Los bonus, explicados

- **Embeddings semánticos**: un modelo pequeño convierte cada fragmento y cada pregunta en un vector numérico que representa su significado. Fragmentos con significado parecido quedan cerca aunque no compartan palabras. Resuelve el problema de las paráfrasis.
- **Búsqueda híbrida**: combinar la lista léxica y la semántica en una sola (una técnica típica es combinar por posiciones en el ranking en vez de por puntuaciones).
- **Indexación incremental**: detectar qué archivos han cambiado (por fecha de modificación o por una huella del contenido) y reindexar solo esos.
- **Caché**: guardar resultados de consultas repetidas y acelerar la carga del índice.
- **API HTTP local**: un pequeño servidor web en tu máquina con rutas para buscar y responder, para usar el sistema desde otros programas.

---

## 13. La defensa

Te pueden pedir modificar algo en directo: cambiar el formato de una salida, añadir un campo a un modelo, cambiar un parámetro de BM25, añadir una opción a un comando. Solo lo harás con soltura si entiendes cada parte, sobre todo si la IA te ha ayudado. Asegúrate de poder explicar con tus palabras cómo troceas, cómo puntúas, cómo calculas el recall y cómo construyes el prompt.
