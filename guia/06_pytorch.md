# PyTorch: qué es y cómo usarlo bien en este proyecto

> Como el resto de la guía, este documento no contiene código. Explica los conceptos y propone ejercicios con soluciones paso a paso.

---

## 1. Qué es PyTorch

**PyTorch** (el paquete se llama `torch`) es una librería de Python para hacer cálculo numérico con **tensores**, sobre todo para redes neuronales. La creó Meta y hoy es el estándar en investigación y en la mayoría de modelos abiertos, incluido Qwen.

Hace tres cosas principales:

1. **Operar con tensores**: tablas de números de cualquier número de dimensiones (una lista, una matriz, un cubo…), con operaciones matemáticas muy rápidas escritas en C++.
2. **Ejecutar esas operaciones en distintos dispositivos**: CPU, GPU NVIDIA (CUDA), GPU AMD, chips de Apple (MPS).
3. **Calcular gradientes automáticamente** (autograd), que es lo que permite **entrenar** redes. En este proyecto **no entrenas nada**, así que esta parte te interesa solo para saber **desactivarla**.

### Cómo encaja con transformers y Qwen

- **Qwen3-0.6B** son unos 600 millones de números (los pesos) guardados en un archivo.
- **transformers** sabe cómo está construida la red de Qwen (capas, atención…). Lee el archivo y crea un objeto modelo de PyTorch con esos pesos dentro.
- **PyTorch** hace los cálculos cuando le pasas un texto convertido en números.

Es decir: `transformers` es la capa cómoda de arriba y `torch` el motor de abajo. En tu código casi todo lo harás con `transformers`, pero vas a tocar `torch` en unos pocos puntos clave, y hacerlo mal cuesta memoria, velocidad o errores raros.

---

## 2. Conceptos que necesitas dominar

### 2.1 Tensor

Es el tipo de dato central. Tiene tres propiedades que siempre debes tener presentes:

- **Forma (shape)**: cuántos elementos tiene en cada dimensión. Cuando tokenizas una frase, obtienes un tensor de forma "1 × número de tokens": la primera dimensión es el **lote** (batch, cuántas frases procesas a la vez) y la segunda los tokens. Muchos errores vienen de olvidar esa dimensión de lote.
- **Tipo de dato (dtype)**: qué clase de número guarda. Enteros (los ids de tokens) o decimales de distinta precisión: 32 bits (precisión completa), 16 bits (la mitad de memoria) o *bfloat16* (16 bits con más rango).
- **Dispositivo (device)**: dónde vive el tensor, en la memoria de la CPU o en la de la GPU.

### 2.2 Dispositivo (device)

Regla de oro: **el modelo y los datos que le pasas deben estar en el mismo dispositivo.** Si el modelo está en GPU y los tokens en CPU, obtienes un error. En tu máquina no hay GPU NVIDIA, así que todo irá en CPU. Aun así, conviene que tu código detecte el dispositivo disponible en vez de fijarlo, porque el evaluador puede tener otra máquina (el subject dice que los bonus corren en CPU, así que CPU debe funcionar siempre).

### 2.3 Tipo de dato y memoria

La memoria que ocupa el modelo es aproximadamente *número de parámetros × bytes por número*:

- 32 bits → 4 bytes por número → unos 2,4 GB para 0,6 mil millones.
- 16 bits → 2 bytes → unos 1,2 GB.

En GPU se suele usar 16 bits. En CPU, históricamente 32 bits era más seguro y a veces más rápido, porque no todas las CPU aceleran 16 bits; *bfloat16* puede funcionar bien en CPUs modernas. **Mide en tu máquina** en lugar de suponer.

### 2.4 Modo evaluación

Los modelos tienen dos modos: entrenamiento y evaluación. Algunas capas se comportan distinto en cada uno. Para usar el modelo (inferencia) hay que ponerlo en **modo evaluación**. `transformers` suele hacerlo al cargar, pero es buena práctica asegurarlo.

### 2.5 Desactivar gradientes

Por defecto, PyTorch **guarda información de cada operación** por si luego quieres entrenar. En inferencia eso es memoria y tiempo tirados. PyTorch tiene **gestores de contexto** (los mismos bloques "with" que el subject pide para recursos) que desactivan esto: uno clásico "sin gradientes" y otro más moderno y algo más rápido, de "modo inferencia". Todo lo que ejecutes con el modelo debería ir dentro de uno de ellos. El método de generación de `transformers` ya lo hace internamente, pero si llamas al modelo directamente (como hacía `llm_sdk` para obtener logits), te toca a ti.

### 2.6 Hilos en CPU

En CPU, PyTorch reparte el trabajo entre varios núcleos. Puedes consultar y ajustar cuántos hilos usa. Normalmente el valor por defecto está bien, pero si notas la máquina colapsada o lentitud rara, es un parámetro a revisar.

### 2.7 Semilla aleatoria

Si generas con aleatoriedad (muestreo con temperatura), cada ejecución da un texto distinto. PyTorch permite fijar una **semilla** para que los resultados sean reproducibles, lo cual ayuda mucho a depurar y comparar prompts. Con generación "voraz" (siempre el token más probable) no hay aleatoriedad.

### 2.8 Instalación: CPU frente a CUDA

El paquete `torch` por defecto en Linux incluye soporte para GPU NVIDIA (CUDA) y pesa varios GB. Si no tienes GPU NVIDIA, existe una versión **solo CPU** mucho más ligera. Con uv se consigue indicándole un índice de paquetes alternativo (el de PyTorch para CPU) en el `pyproject.toml`; la documentación de uv tiene una guía específica de "uso con PyTorch". Ten en cuenta:

- El evaluador solo ejecuta `uv sync`, así que la configuración debe quedar en `pyproject.toml` y `uv.lock`, no en comandos que ejecutaste a mano.
- Si fuerzas la versión CPU, en una máquina con GPU seguirá funcionando (solo que en CPU). Es la opción más segura para este proyecto.

### 2.9 Caché de modelos y espacio en disco

Esto no es de PyTorch, sino de Hugging Face, pero va de la mano: el modelo descargado se guarda en una carpeta de caché en tu directorio personal. Se puede redirigir con una variable de entorno de Hugging Face. En las máquinas de 42, apúntala a `goinfre` para no llenar tu cuota.

### 2.10 Errores típicos

| Síntoma | Causa probable |
|---|---|
| Error diciendo que hay tensores en dispositivos distintos | Modelo y datos en sitios diferentes. |
| Error de forma o dimensiones | Falta (o sobra) la dimensión de lote. |
| Error de tipo de dato | Pasar decimales donde se esperan enteros (ids) o mezclar precisiones. |
| La memoria crece sin parar en un bucle | Inferencia sin desactivar gradientes, o guardar tensores en listas en vez de convertirlos a números normales. |
| Muy lento en CPU | Precisión poco adecuada, modelo recargado en cada pregunta, prompts enormes o demasiados hilos compitiendo. |
| Error de "memoria insuficiente" | Prompt demasiado largo o lotes demasiado grandes. |
| mypy se queja de torch | Tipos de torch complejos; tipa tus funciones con tipos simples de Python en la frontera (texto de entrada, texto de salida). |

---

## 3. Buenas prácticas para este proyecto

1. **Carga el modelo una vez** y reutilízalo para todas las preguntas.
2. **Detecta el dispositivo** automáticamente y mueve modelo y entradas al mismo.
3. **Modo evaluación + inferencia sin gradientes** siempre.
4. **Elige la precisión midiendo**, no suponiendo.
5. **Convierte los resultados a tipos de Python** (texto, listas de enteros) en cuanto puedas; no arrastres tensores por todo tu programa. Así el resto del código no depende de torch y es más fácil de tipar y testear.
6. **Aísla todo lo de torch/transformers en un único módulo** (tu "generador"). El indexador y el buscador no deberían importar torch: así `index` y `search` arrancan rápido y cumplen los tiempos exigidos.
7. **Captura los errores** de carga (sin conexión, sin espacio, sin memoria) y muéstralos con un mensaje claro.
8. **Controla la longitud** de entrada y salida para no quedarte sin memoria ni tiempo.

---

## 4. Ejercicios

Hazlos en una carpeta de prácticas con su propio proyecto uv.

**T1. Instalación.** Crea un proyecto uv e instala `torch` en su versión solo CPU. Comprueba la versión instalada, si detecta GPU y cuánto ocupa el entorno virtual. Compáralo con lo que ocuparía la versión CUDA.

**T2. Primer tensor.** Crea un tensor a partir de una lista de números, otro a partir de una lista de listas y otro lleno de ceros de forma 3 × 4. Imprime la forma, el tipo de dato y el dispositivo de cada uno.

**T3. Operaciones.** Suma, multiplica elemento a elemento y haz el producto matricial de dos tensores. Busca el valor máximo y la posición del máximo de un tensor (esto último es exactamente lo que hace la generación voraz con los logits).

**T4. Formas.** A partir de un tensor de 6 elementos, obtén uno de forma 2 × 3 y otro de 3 × 2. Añade una dimensión de lote al principio (forma 1 × 6) y quítala después.

**T5. Tipos de dato.** Crea un tensor de decimales de 32 bits, conviértelo a 16 bits y a entero. Observa qué se pierde en cada conversión. Calcula cuánta memoria ocupa cada versión.

**T6. De tensor a Python y vuelta.** Convierte un tensor a lista de Python, un tensor de un solo elemento a número de Python, y una lista a tensor.

**T7. Dispositivo.** Escribe una función que devuelva el mejor dispositivo disponible (GPU NVIDIA, Apple o CPU). Crea un tensor directamente en ese dispositivo y otro en CPU, e intenta sumarlos; si tienes solo CPU, razona qué pasaría en una máquina con GPU.

**T8. Gradientes.** Crea un tensor que "requiera gradiente", haz una operación con él y observa que el resultado guarda un historial. Repite la operación dentro del bloque "sin gradientes" y dentro del de "modo inferencia" y compara.

**T9. Semilla.** Genera 5 números aleatorios dos veces sin semilla y dos veces fijando la misma semilla antes. Compara.

**T10. Hilos.** Consulta cuántos hilos usa PyTorch. Multiplica dos matrices grandes (por ejemplo, 2000 × 2000) con 1 hilo, con la mitad de tus núcleos y con todos, midiendo el tiempo.

**T11. Cargar Qwen y mirarlo por dentro.** Carga `Qwen/Qwen3-0.6B` con `transformers`. Averigua: en qué dispositivo está, qué tipo de dato usan sus pesos, cuántos parámetros tiene en total, si está en modo evaluación y cuánta memoria RAM ocupa el proceso.

**T12. Del texto a los logits.** Tokeniza una frase y observa el tensor resultante (forma, tipo). Pásalo al modelo sin gradientes y observa la forma de los logits de salida. Explica qué significa cada dimensión y obtén el token más probable siguiente, convirtiéndolo a texto.

**T13. Precisión en CPU.** Carga el modelo en 32 bits y en bfloat16. Para cada uno, mide el tiempo de carga, la memoria usada y el tiempo en generar 50 tokens para el mismo prompt. Compara también el texto generado. Decide cuál usarás.

**T14. Fuga de memoria.** Haz un bucle de 20 llamadas directas al modelo **sin** desactivar gradientes y guardando la salida en una lista. Mide la memoria. Repite desactivando gradientes y guardando solo el número/texto que te interesa. Compara.

**T15. Longitud del prompt.** Mide el tiempo de generación de 30 tokens con un prompt de 100, 1000 y 3000 tokens. Saca conclusiones sobre cuántos fragmentos de contexto te conviene pasar.

**T16. Módulo generador aislado.** Diseña (en papel y después en código) un módulo que sea el único de tu proyecto que importe torch y transformers, con una interfaz que reciba texto y devuelva texto. Comprueba que `index` y `search` no lo cargan, midiendo su tiempo de arranque con y sin la importación.

**T17. Errores controlados.** Provoca y captura: nombre de modelo inexistente, carga sin conexión con la caché vacía, y un prompt absurdamente largo. En los tres casos debe salir un mensaje claro sin traceback.

---

## 5. Soluciones

**T1.**
1. Crea un proyecto nuevo con uv.
2. Busca en la documentación de uv la guía de integración con PyTorch: explica cómo declarar en `pyproject.toml` el índice de paquetes de PyTorch para CPU y asociar `torch` a ese índice.
3. Añade `torch` y sincroniza.
4. Con un script, imprime la versión de torch y la respuesta de la función que indica si CUDA está disponible (debería ser falso).
5. Mide el tamaño de la carpeta del entorno virtual. Compara con el tamaño aproximado de la versión CUDA (varios GB) que indica la web de PyTorch.
6. Borra el entorno y haz `uv sync`: debe volver a instalar la versión CPU sin que tengas que hacer nada. Eso confirma que funcionará en la evaluación.

**T2.**
1. Usa la función de torch que construye un tensor a partir de datos de Python.
2. Para ceros, usa la función que crea un tensor lleno de ceros indicando la forma.
3. Cada tensor tiene atributos de forma, tipo de dato y dispositivo; imprímelos.
4. Observa que una lista de enteros da un tensor de enteros de 64 bits y una de decimales, decimales de 32 bits.

**T3.**
1. Los operadores aritméticos normales funcionan elemento a elemento entre tensores de la misma forma.
2. El producto matricial tiene su propio operador y su propia función; comprueba que las formas son compatibles (columnas del primero = filas del segundo).
3. Busca las funciones de máximo y de "posición del máximo" (argmax).
4. Relaciónalo con call-me-maybe: allí tu función argmax hacía esto con una lista de Python, mucho más lento que la versión de torch.

**T4.**
1. Crea un tensor de 6 elementos con la función de rangos.
2. Usa la función de cambio de forma para pasarlo a 2 × 3 y a 3 × 2. Comprueba que el producto de dimensiones debe ser siempre 6.
3. Para añadir una dimensión de tamaño 1 en una posición, hay una función específica; para quitar dimensiones de tamaño 1, su inversa.
4. Imprime la forma tras cada paso.

**T5.**
1. Crea un tensor de decimales con varios dígitos.
2. Usa el método de conversión de tipo para pasarlo a 16 bits y a entero.
3. En 16 bits verás pérdida de precisión en los últimos decimales; en entero, se pierden todos los decimales.
4. Cada tensor sabe cuántos bytes ocupa cada elemento y cuántos elementos tiene; multiplica para obtener la memoria. Extrapola al modelo de 600 millones de parámetros.

**T6.**
1. Los tensores tienen un método para convertirse en lista de Python y otro para convertir un tensor de un solo valor en número.
2. Para el camino inverso, usa la misma función de creación de T2.
3. Fíjate que esta conversión es la frontera entre "mundo torch" y "mundo Python" que te recomiendan mantener en un solo sitio.

**T7.**
1. torch tiene funciones para preguntar si hay CUDA y si hay MPS (Apple).
2. Tu función comprueba en ese orden y devuelve CPU si no hay ninguno. Es la misma lógica que tenía `Small_LLM_Model` en call-me-maybe; léela como referencia.
3. Al crear un tensor, puedes indicar el dispositivo, o moverlo después con el método correspondiente.
4. En una máquina con GPU, sumar un tensor en GPU con uno en CPU lanza un error de dispositivos distintos. La solución es moverlos al mismo.

**T8.**
1. Al crear un tensor de decimales, activa la opción de que requiera gradiente.
2. Multiplícalo por 2 y mira el resultado: tiene un atributo que apunta a la función que lo creó (el historial).
3. Repite dentro del gestor de contexto sin gradientes: el resultado ya no tiene historial ni requiere gradiente.
4. Repite con el de modo inferencia: igual, pero el tensor queda marcado como de inferencia.
5. Conclusión: en tu proyecto todo uso del modelo debe ir en uno de estos bloques.

**T9.**
1. Genera números aleatorios con la función de torch dos veces e imprímelos: son distintos.
2. Llama a la función de fijar semilla con un número cualquiera antes de cada generación: salen idénticos.
3. Piensa dónde fijarías la semilla en tu proyecto (al arrancar el generador) y si la harías configurable desde la CLI.

**T10.**
1. Consulta el número de hilos con la función de torch correspondiente y el número de núcleos con la librería estándar.
2. Crea dos matrices aleatorias grandes.
3. Para cada configuración, fija el número de hilos, toma el tiempo, multiplica y vuelve a tomar el tiempo.
4. Normalmente más hilos es más rápido hasta cierto punto; por encima de los núcleos físicos puede empeorar.

**T11.**
1. Carga modelo y tokenizador con las clases Auto de `transformers`.
2. El modelo tiene atributos de dispositivo y tipo de dato; imprímelos.
3. Recorre sus parámetros sumando el número de elementos de cada uno; deberías obtener en torno a 600 millones.
4. El modelo tiene un atributo booleano que indica si está en modo entrenamiento; compruébalo y, si hace falta, ponlo en modo evaluación.
5. Mide la memoria del proceso con una herramienta del sistema o con una librería como psutil antes y después de cargarlo.

**T12.**
1. Tokeniza la frase pidiendo que devuelva tensores de PyTorch.
2. Observa la forma: 1 × número de tokens, y tipo entero.
3. Dentro de un bloque de inferencia, pasa los ids al modelo. La salida tiene un campo de logits.
4. Su forma es 1 × número de tokens × tamaño del vocabulario (unos 150.000): para cada posición, una puntuación para cada token posible como siguiente.
5. Toma la última posición, aplica argmax y decodifica ese id con el tokenizador.
6. Esto es exactamente lo que hacía `get_logits_from_input_ids` + tu bucle en call-me-maybe. El método de generación de `transformers` lo repite de forma eficiente.

**T13.**
1. Carga el modelo indicando el tipo de dato de 32 bits; mide tiempo y memoria.
2. Genera 50 tokens con un prompt fijo y generación voraz; mide el tiempo.
3. Libera ese modelo (borra la referencia) y repite con bfloat16.
4. Compara las tres métricas y la calidad del texto.
5. Anota la decisión y sus números: es material para la sección "Design decisions" del README.

**T14.**
1. Mide la memoria del proceso al inicio.
2. En un bucle, llama al modelo directamente (no al método de generación) sin bloque de inferencia, guarda la salida completa en una lista y mide la memoria en cada vuelta.
3. Repite dentro del bloque de inferencia y guardando solo el token resultante convertido a número de Python.
4. Compara el crecimiento de memoria de ambos casos.

**T15.**
1. Construye tres prompts de distinta longitud repitiendo texto de la documentación de vLLM, y verifica su tamaño en tokens con el tokenizador.
2. Genera 30 tokens con cada uno, midiendo el tiempo.
3. Observa cómo crece el tiempo con la longitud.
4. Calcula cuánto tardarías en responder 100 preguntas con cada longitud y decide un presupuesto de tokens razonable.

**T16.**
1. Define qué necesita saber el resto del programa: "dame una pregunta y unos fragmentos de texto, te devuelvo una respuesta en texto". Esa es la interfaz.
2. Pon en ese módulo la carga del modelo, la plantilla de chat, la generación y la limpieza de la respuesta.
3. Haz que el módulo se importe solo dentro de los comandos `answer` y `answer_dataset`, no al principio del programa.
4. Mide cuánto tarda en arrancar `search` antes y después del cambio. Importar torch cuesta segundos.

**T17.**
1. Intenta cargar un nombre de modelo inventado y anota el tipo de excepción.
2. Vacía temporalmente la caché (o apúntala a una carpeta vacía), desconecta la red y carga: anota la excepción. Investiga la variable de entorno de "modo sin conexión" de Hugging Face.
3. Pasa un prompt enorme y observa qué ocurre (lentitud extrema, aviso de longitud o error de memoria).
4. Envuelve la carga y la generación en bloques de captura con mensajes claros, y controla la longitud antes de generar para que el tercer caso no llegue a producirse.
