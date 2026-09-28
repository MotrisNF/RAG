# Resumen del proyecto: RAG against the machine

## En una frase

Construir un programa en Python que, dada una pregunta sobre el proyecto **vLLM**, encuentre los trozos exactos de su código y documentación que la responden, y use un modelo de lenguaje pequeño (**Qwen3-0.6B**) para redactar una respuesta basada en esos trozos.

## La idea de fondo

Un LLM solo sabe lo que aprendió al entrenarse. En lugar de reentrenarlo, se le "chivan" los textos relevantes justo antes de responder. A eso se le llama **RAG** (Retrieval-Augmented Generation) y tiene cuatro fases:

| Fase | Qué hace |
|---|---|
| **Indexar** | Trocear los archivos de vLLM en fragmentos y guardarlos en un índice consultable. |
| **Recuperar** | Dada una pregunta, encontrar los k fragmentos más relevantes. |
| **Aumentar** | Meter esos fragmentos en el prompt del modelo. |
| **Generar** | El modelo redacta la respuesta usando ese contexto. |

## Qué se evalúa sobre todo

**La búsqueda**, no la redacción. La moulinette mide el **recall@k**: de las fuentes correctas de cada pregunta, cuántas aparecen entre tus k primeros resultados (mismo archivo + rango de caracteres que se solape un poco).

## Requisitos clave

- **Corpus:** el repositorio `vllm-0.10.1`, que debe estar en `data/raw/vllm-0.10.1/`.
- **Troceado:** dos estrategias distintas (una para Python, otra para Markdown/texto). Ningún fragmento puede superar 2000 caracteres.
- **Búsqueda:** obligatorio TF-IDF o BM25.
- **Modelo:** `Qwen/Qwen3-0.6B`, ejecutado en local con la librería `transformers` (sobre `torch`). No se entrega un `llm_sdk` como en call-me-maybe.
- **Objetivos:** recall@5 ≥ 80 % en preguntas de documentación y ≥ 50 % en preguntas de código.
- **Rendimiento:** indexar en ≤ 5 min; buscar 200 preguntas en ≤ 90 s.
- **Herramientas obligatorias:** Python ≥ 3.10, `uv`, `pydantic`, `fire` (CLI), `tqdm` (barras de progreso), `flake8` y `mypy` limpios, docstrings, type hints.
- **Comandos de la CLI:** `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`.
- **Robustez:** nunca debe salir un traceback, ni con consultas vacías, `k=0`, archivos inexistentes o JSON roto.
- **Entregables:** `src/`, `pyproject.toml`, `uv.lock`, `Makefile`, `README.md` (en inglés, con secciones concretas), `.gitignore`. Sin datos, índices ni pesos del modelo en el repo.

## Bonus (solo si lo obligatorio es perfecto)

Embeddings semánticos, búsqueda híbrida, indexación incremental, caché y una API HTTP local.

## Flujo completo

`index` → `search_dataset` → moulinette (puntúa) → `answer_dataset`
