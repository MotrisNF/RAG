## It's a place-holder.

### So much important!!

I know i'm stupid, then don't forget to do this:

- Download the moulinnete. It's only at home.
- Download the rad data. It's only at home.

Please, don't be like always :)./m

---

## Commands (copy & paste)

Run everything from the project root.

### Quick test on the mini corpus (`tests/data`)

```bash
make test_rag
make test_rag TEST_DATASET=dataset_mini_hard.json
make test_rag TEST_DATASET=dataset_mini_hard.json TEST_CHUNK=2000 TEST_K=10
```

### index

```bash
# Defaults: data/raw -> data/processed, chunks of 2000 characters
uv run -m src index

# All options
uv run -m src index --max_chunk_size 2000 --raw_dir data/raw --index_dir data/processed
```

### search

```bash
# Defaults: k = 10, index in data/processed
uv run -m src search "How do I load a LoRA adapter?"

# All options
uv run -m src search "How do I load a LoRA adapter?" --k 5 --index_dir data/processed
```

### search_dataset

```bash
# Only the required option (k = 10, saved to data/output/search_results/UnansweredQuestions)
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json

# All options
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions --index_dir data/processed
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions --index_dir data/processed
```

### evaluate

```bash
# Only the required options (k = 10)
uv run -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
uv run -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json

# All options: --show_failed lists every question not found within k (like the moulinette's list_valid_questions)
uv run -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5 --show_failed
uv run -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --k 5 --show_failed

# Failed questions of the mini hard dataset (run `make test_rag TEST_DATASET=dataset_mini_hard.json` first)
uv run -m src evaluate --student_search_results_path tests/data/output/search_results/dataset_mini_hard.json --dataset_path tests/data/datasets/AnsweredQuestions/dataset_mini_hard.json --k 5 --show_failed
```

### moulinette (official recall, for comparison)

```bash
# Recall@1/3/5/10
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 10 --max_context_length 2000

# PASS/FAIL against the subject thresholds (recall@5: docs 0.8, code 0.5)
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --threshold 0.8
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --threshold 0.5

# [VALID] / [INVALID] for every question (same list as `evaluate --show_failed`)
./moulinette list_valid_questions data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5
./moulinette list_valid_questions data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 5
```
