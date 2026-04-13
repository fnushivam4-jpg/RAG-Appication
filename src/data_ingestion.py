from typing import List
from langchain_core.documents import Document
from langchain_core.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path


class PDFLoader:
	"""PDF loader utility that returns `Document` objects with metadata.

	Usage:
		loader = PDFLoader(loader="pymupdf")
		docs = loader.load("path/to/file.pdf")
		docs_all = loader.load_directory("path/to/folder", recursive=False)
	"""

	def __init__(self, loader: str = "pymupdf") -> None:
		self.loader = loader

	def _create_loader(self, path: Path):
		if self.loader.lower() in ("pymupdf", "fitz", "mupdf"):
			return PyMuPDFLoader(str(path))
		return PyPDFLoader(str(path))

	def load(self, path: str) -> List[Document]:
		"""Load a single PDF and return Documents with added metadata.

		Sets metadata keys: `file_type`, `source`, `file_name`.
		"""
		p = Path(path)
		if not p.exists():
			raise FileNotFoundError(f"PDF not found: {p}")

		pdf_loader = self._create_loader(p)
		docs = pdf_loader.load()

		for d in docs:
			try:
				if not hasattr(d, "metadata") or d.metadata is None:
					d.metadata = {}
			except Exception:
				continue
			d.metadata["file_type"] = "pdf"
			d.metadata.setdefault("source", str(p))
			d.metadata.setdefault("file_name", p.name)

		return docs

	def load_directory(self, dir_path: str, recursive: bool = False) -> List[Document]:
		"""Load all PDF files in a directory (non-PDF files are ignored).

		Args:
			dir_path: directory containing PDFs
			recursive: if True, search subdirectories
		"""
		p = Path(dir_path)
		if not p.exists() or not p.is_dir():
			raise NotADirectoryError(f"Not a directory: {p}")

		pattern = "**/*.pdf" if recursive else "*.pdf"
		all_docs: List[Document] = []
		for f in p.glob(pattern):
			if f.is_file():
				all_docs.extend(self.load(str(f)))

		return all_docs
