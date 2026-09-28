# Ejercicios de práctica

Ejercicios pequeños e independientes para practicar cada pieza del proyecto antes de montarlo todo. Hazlos en una carpeta aparte (por ejemplo, `practicas/`) para no ensuciar el repositorio. Las soluciones están al final, en pasos y sin código.

---

## Bloque A: entorno

**A1. Proyecto con uv.** Crea un proyecto nuevo con uv, añade `pydantic`, `fire` y `tqdm` como dependencias, y ejecuta un script que imprima "hola" usando `uv run`.

**A2. Makefile básico.** Crea un Makefile con las reglas `install`, `run`, `clean` y `lint` para el proyecto del ejercicio A1.

**A3. Linters.** Escribe una función sin type hints ni docstring, pasa flake8 y mypy con los flags del subject, y corrige todo hasta que no haya errores.

## Bloque B: pydantic y JSON

**B1. Primer modelo.** Define un modelo "Persona" con nombre (texto) y edad (entero). Crea una persona válida y otra con edad "abc". Observa qué ocurre.

**B2. Leer un dataset.** Crea a mano un JSON con tres preguntas siguiendo el formato de `RagDataset` (una de ellas sin `question_id`). Léelo y valídalo con los modelos del subject. Comprueba que a la pregunta sin id se le genera uno.

**B3. JSON roto.** Pasa a tu lector del ejercicio B2 un archivo que no existe, un archivo con JSON mal escrito y un JSON válido con un campo que falta. Haz que en los tres casos se muestre un mensaje claro sin traceback.

**B4. Escribir salida.** Construye a mano un objeto `StudentSearchResults` con una pregunta y dos fuentes, y guárdalo en un archivo JSON dentro de una carpeta que todavía no existe.

## Bloque C: CLI con Fire y tqdm

**C1. Calculadora.** Haz una CLI con Fire que tenga los comandos `sumar` y `multiplicar`, cada uno con dos números como opciones.

**C2. Comando con opciones.** Haz un comando `saludar` que reciba un nombre como argumento posicional y una opción `--veces` con valor por defecto 1.

**C3. Módulo ejecutable.** Convierte la CLI de C1 en una carpeta `src` que se ejecute con `python -m src sumar --a 2 --b 3`.

**C4. Barra de progreso.** Recorre todos los archivos `.md` de `vllm-0.10.1/docs` mostrando una barra de progreso, y al final imprime cuántos hay y cuántos caracteres suman.

**C5. Entradas malas.** En la CLI de C1, haz que pasar texto en vez de números, o no pasar nada, muestre un error claro.

## Bloque D: troceado (chunking)

**D1. Troceado ingenuo.** Parte un archivo Markdown de vLLM en trozos de 500 caracteres. Para cada trozo guarda su posición inicial y final. Comprueba que, si recortas el texto original con esas posiciones, obtienes exactamente el trozo.

**D2. Troceado por títulos.** Parte `docs/features/lora.md` por secciones (cada título empieza una sección). Muestra el título y el tamaño de cada sección.

**D3. Respetar el máximo.** Mejora D2 para que ninguna sección supere un tamaño máximo dado; si lo supera, subdivídela por párrafos. Verifica al final que ningún trozo pasa del límite.

**D4. Estructura de un archivo Python.** Con las herramientas de la biblioteca estándar de Python, lista las clases y funciones de `vllm/sampling_params.py` con la línea donde empieza y acaba cada una.

**D5. Troceado de Python.** A partir de D4, crea un trozo por cada función o clase, con posiciones en caracteres (no en líneas). Si una clase supera el máximo, divídela por métodos.

**D6. Rutas correctas.** Recorre la carpeta del corpus y muestra la ruta de 5 archivos tal y como deberían aparecer en la salida (empezando por `data/raw/vllm-0.10.1/`), independientemente de desde dónde ejecutes el programa.

## Bloque E: búsqueda léxica

**E1. Tokenizador.** Crea un tokenizador que pase a minúsculas y separe por cualquier cosa que no sea letra o número. Pruébalo con "How do I enable LoRA adapters?".

**E2. Identificadores.** Mejora el tokenizador para que `max_model_len` produzca "max", "model", "len" y también "max_model_len", y que `LoRARequest` produzca "lora", "request" y "lorarequest".

**E3. TF-IDF a mano.** Con tres frases cortas inventadas, calcula en papel el TF, el IDF y el TF-IDF de dos palabras. Después compruébalo con un programa.

**E4. BM25 con librería.** Usa una librería de BM25 para indexar 10 frases inventadas y buscar una pregunta. Muestra el ranking con puntuaciones.

**E5. Buscador sobre docs.** Junta D3 y E4: indexa todos los `.md` de vLLM troceados y busca "How to configure the OpenAI compatible server?". Muestra los 5 primeros resultados con ruta y posiciones.

**E6. Guardar y cargar el índice.** Guarda el índice de E5 en disco. En otro programa, cárgalo y busca. Mide cuánto tarda cargarlo y cuánto tarda una búsqueda.

**E7. Medir tiempos.** Mide cuánto tardan 200 búsquedas seguidas con el índice cargado una sola vez.

## Bloque F: evaluación

**F1. IoU.** Calcula en papel el IoU de estos pares de rangos: (0–100 con 50–150), (0–100 con 200–300), (0–2000 con 900–950). ¿Cuáles superan 0,05?

**F2. Recall de una pregunta.** Dadas 2 fuentes correctas y 5 recuperadas (invéntalas), calcula el recall@1, @3 y @5.

**F3. Evaluador.** Programa un evaluador que lea un archivo de resultados y un dataset con respuestas y muestre el recall@1, @3, @5 y @10 medio.

## Bloque G: el modelo de lenguaje

> Antes de este bloque, conviene hacer los ejercicios de PyTorch de [06_pytorch.md](06_pytorch.md).

**G1. Comunícate con un LLM.** Carga `Qwen/Qwen3-0.6B` en local y haz que responda a "What is the capital of France?".

**G2. Plantilla de chat.** Repite G1 usando un mensaje de sistema ("Answer in one sentence") y un mensaje de usuario, construyendo el prompt con la plantilla de chat del tokenizador. Imprime el prompt final para ver su forma.

**G3. Modo pensamiento.** Observa si la respuesta de G2 contiene una parte de razonamiento. Desactívala y compara el resultado y el tiempo.

**G4. Contar tokens.** Cuenta cuántos tokens ocupa un fragmento de 2000 caracteres de código y otro de 2000 caracteres de documentación. ¿Son iguales?

**G5. Respuesta con contexto.** Pregúntale al modelo "What is the default port of the vLLM server?" primero sin contexto y después pegándole un fragmento de la documentación que lo diga. Compara.

**G6. Presupuesto.** Dados 10 fragmentos ordenados por relevancia y un máximo de 1500 tokens de contexto, mete en el prompt solo los que quepan, empezando por el más relevante.

## Bloque H: todo junto

**H1. Mini RAG.** Une E5 y G5: una pregunta entra, se buscan 3 fragmentos, se construye el prompt y el modelo responde mostrando las fuentes.

**H2. Casos límite.** Pasa a tu mini RAG: una pregunta vacía, "asdkjh qwe", k=0, k=-3, k=100000. Que ninguno dé traceback.

---

# Soluciones

## Bloque A

**A1.**
1. Instala uv si no lo tienes (consulta su documentación oficial).
2. Crea una carpeta y dentro usa el comando de uv para inicializar un proyecto; verás que crea un `pyproject.toml`.
3. Usa el comando de uv para añadir dependencias, una por una o todas juntas. Observa que se actualizan el `pyproject.toml` y aparece un `uv.lock`.
4. Escribe un script que imprima "hola".
5. Ejecútalo anteponiendo `uv run` a la llamada a Python. Fíjate en que uv crea un entorno virtual automáticamente.
6. Borra el entorno virtual y ejecuta `uv sync`: se reconstruye igual. Eso es lo que hará el evaluador.

**A2.**
1. Crea un archivo llamado exactamente `Makefile`.
2. Cada regla es un nombre seguido de dos puntos y, en las líneas siguientes, los comandos, que deben empezar con un tabulador (no espacios).
3. `install` llama a la sincronización de uv; `run` ejecuta tu script con uv; `clean` borra las carpetas de caché de Python y de mypy; `lint` ejecuta flake8 y mypy con los flags que aparecen en el subject.
4. Prueba cada regla con `make nombre_regla`.

**A3.**
1. Escribe una función que sume dos cosas, sin tipos ni documentación.
2. Ejecuta `make lint` y lee cada error: mypy se quejará de que la función no tiene tipos.
3. Añade el tipo de cada parámetro y del retorno.
4. Añade un docstring explicando qué hace, qué recibe y qué devuelve.
5. Si flake8 se queja de líneas largas o espacios, corrígelos.
6. Repite hasta que no salga nada.

## Bloque B

**B1.**
1. Importa la clase base de modelos de pydantic y define una clase que herede de ella con dos atributos anotados con su tipo.
2. Crea una instancia con valores correctos e imprímela.
3. Crea otra con edad "abc": pydantic lanza un error de validación que dice qué campo falla y por qué.
4. Prueba también edad "30" (texto con número): verás que pydantic lo convierte a entero. Investiga el "modo estricto" si no quieres esa conversión.
5. Envuelve la creación errónea en un bloque de captura de errores e imprime un mensaje amable.

**B2.**
1. Copia del subject los modelos `MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion` y `RagDataset` a un archivo de modelos.
2. Escribe a mano un JSON cuya clave principal sea `rag_questions` con una lista de tres objetos con `question_id` y `question`; a uno quítale el id.
3. Abre el archivo con un gestor de contexto, cárgalo como JSON y pásalo al método de validación del modelo `RagDataset`.
4. Recorre las preguntas e imprime su id: la que no lo tenía ahora tiene un identificador aleatorio largo.
5. Pydantic también tiene un método que valida directamente desde texto JSON; pruébalo como alternativa.

**B3.**
1. Identifica qué error distinto se produce en cada caso: archivo inexistente, JSON mal escrito y validación fallida son tres tipos de excepción diferentes.
2. Provócalos uno a uno y anota el nombre de cada excepción.
3. Captura cada una por separado con un mensaje específico ("no existe el archivo X", "el JSON no es válido en la línea N", "falta el campo question").
4. Comprueba que el programa termina de forma ordenada, idealmente con un código de salida distinto de cero.

**B4.**
1. Crea dos `MinimalSource` con rutas y posiciones inventadas.
2. Crea un `MinimalSearchResults` con un id, una pregunta y esas dos fuentes.
3. Crea un `StudentSearchResults` con una lista que contenga ese resultado y un k.
4. Crea la carpeta de destino, pidiendo que se creen también las carpetas intermedias y que no falle si ya existe.
5. Usa el método de pydantic que convierte el modelo a JSON (con sangría para que sea legible) y escríbelo en el archivo con un gestor de contexto.
6. Abre el archivo y compáralo con el ejemplo del subject.

## Bloque C

**C1.**
1. Instala Fire (ya lo hiciste en A1).
2. Crea una clase con dos métodos, cada uno con dos parámetros tipados que devuelvan el resultado.
3. En el punto de entrada, pásale la clase a Fire.
4. Ejecuta el script con el nombre del método y las opciones con doble guion.
5. Ejecuta el script sin argumentos o con `--help` y observa la ayuda que genera Fire automáticamente a partir de los docstrings.

**C2.**
1. Añade un método con un parámetro sin valor por defecto (nombre) y otro con valor por defecto 1 (veces).
2. Ejecútalo pasando el nombre directamente tras el comando: Fire lo trata como posicional.
3. Ejecútalo añadiendo la opción de veces.
4. Fíjate que así es como el subject define `search <query> --k <int>`.

**C3.**
1. Crea una carpeta `src`.
2. Dentro, crea el archivo que marca la carpeta como paquete (puede estar vacío).
3. Crea el archivo especial que Python ejecuta cuando se lanza un paquete con `-m`; ahí va la llamada a Fire.
4. Mueve la clase de C1 a otro archivo del paquete e impórtala desde ese punto de entrada.
5. Desde la carpeta padre de `src`, ejecuta `uv run python -m src sumar ...`.

**C4.**
1. Usa la librería estándar de rutas para buscar recursivamente los archivos con extensión `.md`.
2. Convierte el resultado en lista para saber cuántos hay (tqdm lo necesita para calcular el porcentaje).
3. Envuelve el bucle con tqdm, añadiendo una descripción.
4. Dentro, lee cada archivo como texto UTF-8 y suma su longitud.
5. Imprime el total al acabar. Debería rondar las 178 páginas de Markdown en todo el repo, menos si solo miras `docs`.

**C5.**
1. Prueba primero qué hace Fire con texto en lugar de números: verás que no siempre falla, porque intenta interpretar los valores.
2. Añade al principio de cada método una comprobación de que los valores son números; si no, muestra un error y sal.
3. Haz lo mismo para valores ausentes.
4. Añade una captura general alrededor de la llamada a Fire para que nada inesperado genere un traceback.

## Bloque D

**D1.**
1. Lee el archivo completo como texto.
2. Recorre el texto avanzando de 500 en 500 posiciones.
3. Para cada paso, guarda la posición inicial, la final y el texto de ese tramo (por ejemplo, en una lista de diccionarios o en un pequeño modelo).
4. Comprobación: para cada trozo, vuelve a recortar el texto original con sus posiciones y compara con el texto guardado. Deben ser idénticos.
5. Decide y anota si tu posición final es "inclusiva" o "exclusiva" (si el último carácter está incluido o no). Sé coherente en todo el proyecto.

**D2.**
1. Lee `lora.md` como texto y recórrelo línea a línea, llevando la cuenta de la posición en caracteres donde empieza cada línea (sumando la longitud de cada línea y su salto).
2. Cada vez que una línea empiece por almohadilla, es un título: cierra la sección anterior y abre una nueva.
3. Ten cuidado con los bloques de código: dentro de ellos una línea con almohadilla puede ser un comentario, no un título. Lleva un indicador de "estoy dentro de un bloque de código".
4. Al final, cierra la última sección.
5. Imprime título y tamaño de cada una.

**D3.**
1. Para cada sección de D2, si cabe en el máximo, déjala tal cual.
2. Si no cabe, divídela por líneas en blanco (párrafos) y ve acumulando párrafos mientras quepan; cuando el siguiente no quepa, cierra el trozo y empieza otro.
3. Si un único párrafo ya supera el máximo, córtalo por líneas y, en último caso, por caracteres.
4. Recalcula bien las posiciones de cada subtrozo respecto al archivo original.
5. Al final, recorre todos los trozos y verifica que ninguno supera el máximo y que las posiciones recortan el texto correcto.

**D4.**
1. Busca en la documentación de Python el módulo de la biblioteca estándar que analiza código fuente y produce su árbol sintáctico.
2. Lee el archivo como texto y pásalo a la función de análisis de ese módulo.
3. Recorre los elementos de primer nivel del árbol; los que sean definiciones de función o de clase tienen nombre, línea de inicio y línea de fin.
4. Para las clases, recorre también sus elementos internos para listar los métodos.
5. Ten en cuenta que los decoradores empiezan antes que la línea de la definición; investiga cómo obtener su línea.

**D5.**
1. Calcula una tabla con la posición en caracteres de inicio de cada línea del archivo.
2. Con esa tabla, convierte la línea de inicio y de fin de cada función o clase en posiciones de caracteres.
3. Crea un trozo por cada una; si una clase excede el máximo, crea en su lugar un trozo por cada método (y otro para la cabecera de la clase con su docstring).
4. No olvides el código entre definiciones (importaciones, constantes): agrúpalo en trozos propios.
5. Verifica que ningún trozo excede el máximo y que las posiciones recortan el texto correcto.
6. Prueba qué pasa con un archivo con errores de sintaxis y prepara un plan B (troceado por líneas).

**D6.**
1. Mueve primero `vllm-0.10.1` a `data/raw/`.
2. Recorre `data/raw` recursivamente.
3. Para cada archivo, calcula su ruta relativa respecto a la raíz del proyecto, no respecto a la carpeta desde donde se ejecuta.
4. Comprueba que usa barras normales y que empieza por `data/raw/vllm-0.10.1/`.
5. Ejecuta el programa desde otra carpeta y comprueba que sigue dando lo mismo (o decide que siempre se ejecuta desde la raíz, que es lo que hace el evaluador, y documéntalo).

## Bloque E

**E1.**
1. Pasa el texto a minúsculas.
2. Usa una expresión regular que encuentre secuencias de letras y números.
3. Devuelve la lista de coincidencias.
4. Resultado esperado: how, do, i, enable, lora, adapters.

**E2.**
1. Antes de pasar a minúsculas, detecta cada palabra completa, incluyendo guiones bajos.
2. Guarda la palabra completa en minúsculas.
3. Divídela además por guiones bajos.
4. Divide cada pieza por cambios de minúscula a mayúscula (y por grupos de mayúsculas seguidos de mayúscula+minúscula, para casos como "LoRARequest").
5. Añade todas las piezas en minúsculas a la lista.
6. Prueba con varios identificadores de vLLM y revisa los casos raros.

**E3.**
1. Escribe tres frases de 4–5 palabras donde una palabra salga en las tres y otra solo en una.
2. TF: cuenta cuántas veces sale cada palabra en cada frase (puedes dividir por el total de palabras de la frase).
3. IDF: logaritmo del número de frases entre el número de frases que contienen la palabra. La palabra que sale en todas tendrá IDF cero (o casi, según la variante).
4. TF-IDF = TF × IDF.
5. Verifica con un programa o con una librería. Si las cifras difieren un poco, busca qué variante de fórmula usa la librería (suavizados).

**E4.**
1. Instala una librería de BM25 con uv (hay varias; elige una mantenida y rápida).
2. Tokeniza cada frase con tu tokenizador de E2.
3. Crea el índice pasándole la lista de frases tokenizadas.
4. Tokeniza la pregunta con el mismo tokenizador.
5. Pide las puntuaciones, ordénalas de mayor a menor y muestra las frases con su puntuación.
6. Cambia los parámetros k1 y b y observa cómo cambia el ranking.

**E5.**
1. Recorre todos los `.md` de `data/raw` y trocéalos con D3, guardando ruta y posiciones.
2. Tokeniza el texto de cada trozo (prueba también a añadir la ruta del archivo y el título de la sección al texto que tokenizas).
3. Construye el índice BM25 con todos los trozos.
4. Tokeniza la pregunta, puntúa, ordena y quédate con los 5 primeros.
5. Muestra ruta y posiciones. Deberías ver `docs/serving/openai_compatible_server.md` arriba.

**E6.**
1. Decide qué necesitas guardar: la lista de trozos (ruta, posiciones, texto) y el propio índice.
2. Elige un formato de guardado (serialización binaria de Python, JSON, o el propio de la librería).
3. Guárdalo en una carpeta tipo `data/processed`.
4. En otro programa, mide el tiempo antes y después de cargarlo.
5. Mide el tiempo de una búsqueda por separado.
6. Si la carga es muy lenta, investiga formatos más eficientes.

**E7.**
1. Prepara una lista de 200 preguntas (puedes repetir unas pocas).
2. Carga el índice una vez.
3. Mide el tiempo total de las 200 búsquedas con una barra de progreso.
4. Si supera 90 segundos, busca el cuello de botella: ¿recargas algo en cada búsqueda?, ¿ordenas todo cuando solo necesitas los k mejores?

## Bloque F

**F1.**
1. Para cada par, la intersección es la parte común: desde el mayor de los inicios hasta el menor de los finales (si es negativa, es cero).
2. La unión es la longitud de uno más la del otro menos la intersección.
3. IoU = intersección / unión.
4. Resultados: el primero es 50/150 ≈ 0,33 (supera); el segundo es 0 (no); el tercero es 50/2000 = 0,025 (no supera, aunque el trozo contiene la fuente entera). Esta última es la lección importante.

**F2.**
1. Marca cuáles de tus 5 resultados aciertan alguna fuente correcta (mismo archivo + IoU ≥ 0,05).
2. Para @1, mira solo el primero; para @3, los tres primeros; para @5, los cinco.
3. En cada caso, cuenta cuántas fuentes correctas distintas has cubierto y divide entre 2.
4. Ten en cuenta que dos resultados tuyos que aciertan la misma fuente cuentan una sola vez.

**F3.**
1. Carga el archivo de resultados con `StudentSearchResults` y el de respuestas con `RagDataset`.
2. Crea un diccionario de id de pregunta → fuentes correctas.
3. Para cada pregunta de tus resultados, busca sus fuentes correctas y calcula el recall para cada k como en F2.
4. Haz la media por k sobre todas las preguntas.
5. Gestiona preguntas que no están en el dataset de referencia o sin fuentes correctas.
6. Compara tus números con los de la moulinette para validar tu evaluador.

## Bloque G

**G1.**
1. Añade con uv la librería `transformers` y PyTorch (`torch`), en su versión de CPU si no tienes GPU compatible, para ahorrar espacio. No uses el `llm_sdk` de call-me-maybe: aquí no se proporciona y su generación token a token es demasiado lenta para prompts largos (ver sección 7.1 de la explicación detallada).
2. Carga el tokenizador y el modelo con las clases "Auto" de `transformers`, indicando el nombre `Qwen/Qwen3-0.6B`. Son las mismas que usa `llm_sdk` por dentro. La primera vez se descargará (algo más de 1 GB) en la caché de Hugging Face.
3. Convierte la pregunta en tokens con el tokenizador.
4. Llama al método de generación del modelo (no hagas tú el bucle como en call-me-maybe), limitando el número de tokens nuevos (por ejemplo, 100).
5. Convierte los tokens generados de vuelta a texto, quedándote solo con los nuevos (no con los de la pregunta).
6. Imprime la respuesta.

**G2.**
1. Construye una lista de mensajes: uno con rol de sistema y otro con rol de usuario.
2. Usa la función del tokenizador que aplica la plantilla de chat, pidiéndole que añada el marcador de "ahora responde el asistente".
3. Imprime el texto resultante: verás marcas especiales que separan los roles.
4. Tokeniza ese texto y genera como en G1.

**G3.**
1. Mira si la respuesta contiene un bloque de razonamiento delimitado por etiquetas especiales.
2. Consulta la ficha del modelo en Hugging Face: explica cómo desactivar el modo pensamiento al aplicar la plantilla de chat.
3. Desactívalo y vuelve a generar.
4. Mide el tiempo en ambos casos y compara la longitud y calidad de la respuesta.
5. Como red de seguridad, piensa cómo eliminar ese bloque de la salida si aparece igualmente.

**G4.**
1. Recorta 2000 caracteres de un archivo `.py` y 2000 de un `.md`.
2. Pásalos por el tokenizador y cuenta los tokens.
3. Normalmente el código produce más tokens por carácter (símbolos, sangrías, nombres raros).
4. Saca la conclusión: con k=10 fragmentos de 2000 caracteres, ¿cuántos tokens tendrías? Eso justifica el ejercicio G6.

**G5.**
1. Hazle la pregunta sin contexto y apunta la respuesta.
2. Busca en `docs` la línea donde se menciona el puerto por defecto del servidor.
3. Construye un mensaje de sistema que le pida responder solo con el contexto dado.
4. En el mensaje de usuario pon primero el fragmento (indicando de qué archivo viene) y después la pregunta.
5. Compara ambas respuestas. Prueba también a darle un fragmento que no tenga la respuesta y observa si inventa o admite que no lo sabe; ajusta las instrucciones.

**G6.**
1. Calcula primero cuántos tokens ocupan las instrucciones y la pregunta; réstalos del presupuesto.
2. Recorre los fragmentos por orden de relevancia.
3. Cuenta los tokens de cada uno; si cabe, añádelo y resta su tamaño del presupuesto.
4. Si no cabe, decide: saltarlo y probar con el siguiente, o recortarlo para que quepa.
5. Detente cuando no quede presupuesto.
6. Imprime cuántos fragmentos han entrado.

## Bloque H

**H1.**
1. Carga el índice de E6 y el modelo de G1 una sola vez.
2. Recibe la pregunta y busca los 3 mejores fragmentos.
3. Recupera su texto a partir de ruta y posiciones.
4. Construye el prompt con G5 y G6.
5. Genera la respuesta y muéstrala junto a la lista de fuentes.
6. Prueba con 5 preguntas distintas y valora si las respuestas se apoyan en las fuentes.

**H2.**
1. Añade una validación inicial de la pregunta: si está vacía tras quitar espacios, mensaje y salida.
2. Valida k: debe ser un entero mayor que cero; si es mayor que el número de fragmentos, limítalo.
3. Para la pregunta sin sentido, comprueba qué pasa si ningún término existe en el índice: todas las puntuaciones serán cero. Decide si devuelves resultados igualmente o un mensaje de "sin resultados", pero sin romper.
4. Envuelve toda la ejecución en una captura general.
5. Ejecuta los cinco casos y confirma que ninguno muestra un traceback.
