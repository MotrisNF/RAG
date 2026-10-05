import time
from sentence_transformers import SentenceTransformer
from src.reader import CorpusReader
from src.chunker import PyChunker, MdChunker
import numpy as np

docs = CorpusReader("data/raw").read().documents
chunks = (PyChunker(2000).chunk_documents([d for d in docs if d.file_path.endswith(".py")])
          + MdChunker(2000).chunk_documents([d for d in docs if not d.file_path.endswith(".py")]))

# Chosen after measuring 5 models (README → Semantic embeddings): as good as
# MiniLM at 128 tokens, and the whole corpus in ~2 s instead of ~6 min.
model = SentenceTransformer("minishlab/potion-base-8M", device="cpu")
start = time.time()
vectors = model.encode([c.text for c in chunks], batch_size=512,
                       normalize_embeddings=True, show_progress_bar=True)
print(len(chunks), "chunks in", round(time.time() - start), "s; shape", vectors.shape)
np.save("data/processed/vectors.npy", vectors)
