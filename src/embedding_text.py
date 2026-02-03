from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Literal


DEFAULT_MODEL_NAME = "BAAI/bge-base-zh-v1.5"


def _collapse_whitespace(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def markdown_to_text(markdown: str) -> str:
	"""Best-effort Markdown -> plain text.

	No third-party Markdown parser required; this is intentionally lightweight.
	"""

	text = markdown.replace("\r\n", "\n")

	# Remove BOM if present
	text = text.lstrip("\ufeff")

	# Strip YAML front-matter
	text = re.sub(r"\A---\n[\s\S]*?\n---\n", "", text)

	# Remove fenced code blocks entirely
	text = re.sub(r"```[\s\S]*?```", " ", text)
	text = re.sub(r"~~~[\s\S]*?~~~", " ", text)

	# Inline code
	text = re.sub(r"`([^`]+)`", r"\\1", text)

	# Images: ![alt](url) -> alt
	text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\\1", text)
	# Links: [text](url) -> text
	text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\\1", text)

	# Headings / blockquotes / list markers
	text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
	text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
	text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.MULTILINE)
	text = re.sub(r"^\s{0,3}\d+\.\s+", "", text, flags=re.MULTILINE)

	# Tables / separators
	text = re.sub(r"^\s{0,3}[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)
	text = text.replace("|", " ")

	# Emphasis markers
	text = text.replace("**", " ").replace("__", " ")
	text = text.replace("*", " ").replace("_", " ")
	text = text.replace("~~", " ")

	# Normalize whitespace but preserve paragraph breaks
	text = re.sub(r"[ \t]+", " ", text)
	text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
	text = re.sub(r"\n{3,}", "\n\n", text)
	return text.strip()


@dataclass(frozen=True)
class ChunkingConfig:
	max_tokens: int = 512
	overlap_tokens: int = 64
	min_chunk_chars: int = 20


class TextEmbedder:
	def __init__(
		self,
		model_name: str = DEFAULT_MODEL_NAME,
		device: Literal["auto", "cpu", "cuda"] = "auto",
		batch_size: int = 16,
		normalize: bool = True,
	) -> None:
		self.model_name = model_name
		self.device = device
		self.batch_size = batch_size
		self.normalize = normalize

		self._tokenizer = None
		self._model = None
		self._torch = None

	def _lazy_load(self) -> None:
		if self._model is not None:
			return

		try:
			import torch  # type: ignore
			from transformers import AutoModel, AutoTokenizer  # type: ignore
		except Exception as exc:  # pragma: no cover
			raise RuntimeError(
				"Missing runtime deps for embeddings. Install with: `uv add torch transformers`"
			) from exc

		self._torch = torch
		self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		self._model = AutoModel.from_pretrained(self.model_name)

		if self.device == "auto":
			chosen = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			chosen = self.device

		self._model.to(chosen)
		self._model.eval()

	@property
	def tokenizer(self):
		self._lazy_load()
		return self._tokenizer

	def _mean_pool(self, last_hidden_state, attention_mask):
		torch = self._torch
		mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
		summed = (last_hidden_state * mask).sum(dim=1)
		counts = mask.sum(dim=1).clamp(min=1e-9)
		pooled = summed / counts
		if self.normalize:
			pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
		return pooled

	def embed_texts(self, texts: list[str]) -> list[list[float]]:
		"""Embed a list of texts into vectors."""

		if not texts:
			return []

		self._lazy_load()
		torch = self._torch
		model = self._model
		tokenizer = self._tokenizer

		if self.device == "auto":
			chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			chosen_device = self.device

		vectors: list[list[float]] = []

		with torch.no_grad():
			for start in range(0, len(texts), self.batch_size):
				batch = texts[start : start + self.batch_size]
				encoded = tokenizer(
					batch,
					padding=True,
					truncation=True,
					max_length=512,
					return_tensors="pt",
				)
				encoded = {k: v.to(chosen_device) for k, v in encoded.items()}
				outputs = model(**encoded)
				pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
				vectors.extend(pooled.detach().cpu().tolist())

		return vectors

	def chunk_text(self, text: str, config: ChunkingConfig = ChunkingConfig()) -> list[str]:
		"""Token-aware chunking using the model tokenizer."""
		self._lazy_load()
		tokenizer = self._tokenizer

		if not text or not text.strip():
			return []

		normalized = text.replace("\r\n", "\n")
		paragraphs = [p.strip() for p in re.split(r"\n\n+", normalized)]
		paragraphs = [p for p in paragraphs if p]

		# Fallback: if we lost paragraph boundaries (e.g., single huge paragraph), split by sentences.
		if len(paragraphs) <= 1:
			sentences = re.split(r"(?<=[。！？!?])\s*", _collapse_whitespace(normalized))
			paragraphs = [s.strip() for s in sentences if s and s.strip()]

		chunks: list[str] = []
		current_parts: list[str] = []
		current_tokens = 0

		def flush() -> None:
			nonlocal current_parts, current_tokens
			if not current_parts:
				return
			chunk = _collapse_whitespace("\n\n".join(current_parts))
			if len(chunk) >= config.min_chunk_chars:
				chunks.append(chunk)
			current_parts = []
			current_tokens = 0

		def para_token_len(para: str) -> int:
			return len(tokenizer(para, add_special_tokens=False, verbose=False).input_ids)

		for para in paragraphs:
			ptok = para_token_len(para)
			if ptok <= config.max_tokens:
				if current_tokens and current_tokens + ptok > config.max_tokens:
					flush()
				current_parts.append(para)
				current_tokens += ptok
				continue

			# Paragraph too large: split by sliding window over tokens
			flush()
			ids = tokenizer(para, add_special_tokens=False, verbose=False).input_ids
			step = max(1, config.max_tokens - config.overlap_tokens)
			for i in range(0, len(ids), step):
				window = ids[i : i + config.max_tokens]
				if not window:
					break
				piece = tokenizer.decode(window, skip_special_tokens=True)
				piece = _collapse_whitespace(piece)
				if len(piece) >= config.min_chunk_chars:
					chunks.append(piece)

		flush()
		return chunks


def embed_markdown(
	markdown: str,
	embedder: TextEmbedder | None = None,
	chunking: ChunkingConfig = ChunkingConfig(),
) -> dict:
	"""Embed Markdown content.

	Returns a dict with `chunks` and `vectors`.
	"""

	embedder = embedder or TextEmbedder()
	plain = markdown_to_text(markdown)
	chunks = embedder.chunk_text(plain, config=chunking)
	vectors = embedder.embed_texts(chunks)
	return {"chunks": chunks, "vectors": vectors, "model": embedder.model_name}


def embed_markdown_file(
	path: str | Path,
	embedder: TextEmbedder | None = None,
	encoding: str = "utf-8",
	chunking: ChunkingConfig = ChunkingConfig(),
) -> dict:
	p = Path(path)
	markdown = p.read_text(encoding=encoding, errors="ignore")
	result = embed_markdown(markdown, embedder=embedder, chunking=chunking)
	result["path"] = str(p)
	return result


def embed_inputs_folder(
	inputs_dir: str | Path = "inputs",
	embedder: TextEmbedder | None = None,
) -> list[dict]:
	"""Convenience helper: embed all .md files under inputs/."""
	embedder = embedder or TextEmbedder()
	root = Path(inputs_dir)
	results: list[dict] = []
	for md in sorted(root.glob("**/*.md")):
		results.append(embed_markdown_file(md, embedder=embedder))
	return results


if __name__ == "__main__":
	# Quick smoke-run: embed inputs/*.md and print basic stats
	results = embed_inputs_folder("inputs")
	for r in results:
		dims = len(r["vectors"][0]) if r["vectors"] else 0
		print(f"{r['path']}: chunks={len(r['chunks'])}, dim={dims}, model={r['model']}")