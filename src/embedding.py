from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.text_splitter import RecursiveCharacterTextSplitter


class Embedding:
	"""Utility for chunking PDF Documents prior to embedding.

	Example:
		from notebook.embedding import Embedding
		chunker = Embedding(chunk_size=1000, chunk_overlap=200)
		chunks = chunker.chunk_documents(docs)
	"""

	def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None) -> None:
		"""Create a chunker.

		Args:
			chunk_size: maximum characters per chunk.
			chunk_overlap: overlap between chunks.
			separators: list of preferred separators (e.g. ['\n\n','\n']).
				If None, defaults to ['\n\n', '\n'] which preserves paragraphs.
		"""
		if separators is None:
			separators = ["\n\n", "\n", " ", ""]
		self.splitter = RecursiveCharacterTextSplitter(
			chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
		)

	def chunk_documents(self, docs: List[Document]) -> List[Document]:
		"""Split a list of `Document` objects into smaller chunk `Document`s.

		Preserves and extends metadata with `chunk` (index), and keeps
		`file_type`, `source`, and `file_name` if present.
		"""
		out: List[Document] = []
		# prefer splitter.split_documents when available (preserves splitter behavior)
		split_documents_fn = getattr(self.splitter, "split_documents", None)
		for d in docs:
			text = getattr(d, "page_content", None) or getattr(d, "content", "")
			if not text:
				continue

			if callable(split_documents_fn):
				# split_documents usually accepts a list of Documents and returns Documents
				try:
					split_docs = split_documents_fn([d])
				except Exception:
					# fallback to split_text on any unexpected error
					split_docs = None
			else:
				split_docs = None

			if split_docs:
				for idx, sd in enumerate(split_docs):
					pc = getattr(sd, "page_content", None) or getattr(sd, "content", "")
					md = dict(sd.metadata) if getattr(sd, "metadata", None) else {}
					md.setdefault("file_type", "pdf")
					md.setdefault("source", d.metadata.get("source") if getattr(d, "metadata", None) else None)
					md.setdefault("file_name", d.metadata.get("file_name") if getattr(d, "metadata", None) else None)
					md["chunk"] = idx
					out.append(Document(page_content=pc, metadata=md))
			else:
				# fallback: split_text then construct Documents
				parts = self.splitter.split_text(text)
				for idx, part in enumerate(parts):
					md = dict(d.metadata) if getattr(d, "metadata", None) else {}
					md.setdefault("file_type", "pdf")
					md.setdefault("source", str(md.get("source") or d.metadata.get("source") if getattr(d, "metadata", None) else None))
					md.setdefault("file_name", md.get("file_name") or (d.metadata.get("file_name") if getattr(d, "metadata", None) else None))
					md["chunk"] = idx
					out.append(Document(page_content=part, metadata=md))

		return out

	def embed_documents(self, docs: List[Document], embedder=None, model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
		"""Compute embeddings for a list of chunk `Document`s.

		Args:
			docs: Documents to embed (uses `page_content` or `content`).
			embedder: Optional callable that accepts `List[str]` and returns `List[List[float]]`.
				If provided, it will be called as `embedder(texts)`.
			model_name: sentence-transformers model name used when no embedder is provided.

		Returns:
			List of embedding vectors (lists of floats) in the same order as `docs`.

		Behavior:
			- If `embedder` is a callable, it will be used directly.
			- Otherwise, attempts to use `sentence_transformers.SentenceTransformer`.
		"""
		texts = [getattr(d, "page_content", None) or getattr(d, "content", "") for d in docs]
		# use provided callable embedder
		if callable(embedder):
			return embedder(texts)

		# fallback to sentence-transformers if available
		try:
			from sentence_transformers import SentenceTransformer
		except Exception as e:
			raise ImportError(
				"No embedder provided and sentence-transformers is not installed."
				" Install with `pip install sentence-transformers` or pass a callable embedder."
			) from e

		model = SentenceTransformer(model_name)
		embs = model.encode(texts, show_progress_bar=False)

		# ensure output is a list of lists of floats
		out: List[List[float]] = []
		for v in embs:
			try:
				out.append(list(map(float, v.tolist() if hasattr(v, "tolist") else v)))
			except Exception:
				# last resort: coerce via list()
				out.append([float(x) for x in list(v)])

		return out
