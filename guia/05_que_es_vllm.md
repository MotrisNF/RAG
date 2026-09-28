# ¿Qué es la carpeta `vllm-0.10.1` y para qué la necesito?

## Qué es vLLM

**vLLM** es un proyecto de código abierto muy conocido que sirve para **ejecutar modelos de lenguaje (LLM) de forma rápida y eficiente**, sobre todo en servidores con GPU. Es lo que usan muchas empresas para "servir" modelos: recibir peticiones de muchos usuarios a la vez y devolver respuestas del modelo lo más rápido posible. Incluso ofrece un servidor compatible con la API de OpenAI.

La carpeta que te has descargado es el **código fuente completo de la versión 0.10.1** de ese proyecto: unos 38 MB, cerca de 2.900 archivos, de los cuales unos 1.760 son archivos Python (`.py`) y unos 180 son documentación Markdown (`.md`). Dentro verás, entre otras cosas:

- `vllm/` — el código Python del propio vLLM (motor, servidor, LoRA, cuantización, etc.).
- `docs/` — la documentación en Markdown (guías de uso, configuración, despliegue, funcionalidades…).
- `examples/`, `tests/`, `benchmarks/` — ejemplos, tests y pruebas de rendimiento.
- `csrc/`, `cmake/` — código C++/CUDA de bajo nivel.

## Para qué la necesitas en este proyecto

**No vas a usar vLLM como herramienta. No tienes que instalarlo, ni importarlo, ni ejecutarlo.**

vLLM es **el corpus**: el "libro" sobre el que tu sistema RAG tiene que responder preguntas. Es decir, es el **dato de entrada** de tu proyecto, igual que si fuera una enciclopedia.

- Tu comando `index` leerá sus archivos (sobre todo los `.py` y los `.md`), los troceará y construirá el índice.
- Las preguntas de los datasets serán del estilo "¿Cómo configuro el servidor OpenAI?" o "¿Qué hace tal clase?", y la respuesta estará escondida en algún archivo de esta carpeta.
- Los resultados que devuelvas apuntarán a rutas dentro de esta carpeta, por ejemplo `data/raw/vllm-0.10.1/docs/features/lora.md`.

## Dónde debes ponerla

El subject exige esta estructura, y la moulinette compara las rutas **letra por letra**:

```
<raíz de tu repositorio>/
└── data/
    └── raw/
        └── vllm-0.10.1/     ← aquí
            ├── docs/
            ├── vllm/
            └── ...
```

Ahora mismo la tienes directamente en la raíz del proyecto (`RAG/vllm-0.10.1`). Tendrás que **moverla a `data/raw/vllm-0.10.1`**. Si no, todas las rutas que genere tu sistema tendrán un prefijo distinto y **el recall será 0**, aunque encuentres el fragmento correcto.

## Cosas a tener en cuenta

- **No la subas a tu repositorio Git.** El subject prohíbe incluir datos grandes. Añade `data/raw/` (y `data/processed/`, `data/output/`) al `.gitignore`. El evaluador pondrá su propia copia.
- **No la modifiques.** Los índices de caracteres de las respuestas correctas se calcularon sobre los archivos originales. Si cambias un solo carácter, los rangos dejan de cuadrar.
- **Decide qué archivos indexar.** No tiene sentido indexar imágenes, código C++ o archivos binarios. Lo razonable es empezar por `.py` y `.md` (son justo las dos estrategias de troceado que te piden) y quizá algún otro de texto si ves que mejora el recall.

## Lo que probablemente te falta

El subject menciona archivos que no están en tu carpeta ahora mismo. Búscalos en la página del proyecto en la intra (adjuntos):

- **Los datasets de preguntas**: van en `data/datasets/UnansweredQuestions/` y `data/datasets/AnsweredQuestions/` (p. ej. `dataset_docs_public.json` y su equivalente de código).
- **El ejecutable moulinette** (`moulinette-ubuntu` o `moulinette-fedora`), que se renombra a `moulinette` y sirve para calcular tu recall oficial.
