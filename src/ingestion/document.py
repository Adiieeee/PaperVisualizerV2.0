import os
import fitz  # PyMuPDF
from typing import List, Dict

from src.config import SECTION_FILTER_PATTERN, SECTION_FILTER_MIN_POSITION, MAX_PDF_PAGES, MAX_PDF_BYTES

# PDF text often contains Unicode ligature glyphs (ﬁ, ﬂ, …) that pollute entity names
# (e.g. "classiﬁcation"). Map them back to plain ASCII. General — any corpus, any domain.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}


def _fix_ligatures(text: str) -> str:
    for lig, repl in _LIGATURES.items():
        if lig in text:
            text = text.replace(lig, repl)
    return text


class DocumentProcessor:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _filter_sections(self, text: str) -> str:
        """
        Truncate the document at the first terminal section heading
        (References / Bibliography / Acknowledgements). Everything after it — the citation
        list and any appendix that follows — is dropped: it is a major source of noise
        (citation authors, bibliography entries) and of wasted LLM calls.

        `SECTION_FILTER_MIN_POSITION` guards against a stray early match deleting body
        content. It is deliberately low (0.20) because modern papers with long appendices
        put References at only ~30-47% of the document.
        Applied before chunking, so neither extraction nor the vector store ingest it.
        """
        cut_at = None
        for m in SECTION_FILTER_PATTERN.finditer(text):
            if m.start() >= len(text) * SECTION_FILTER_MIN_POSITION:
                cut_at = m.start()
                break
        if cut_at is not None:
            return text[:cut_at]
        return text

    def process_pdf(self, file_path: str) -> List[Dict[str, str]]:
        """
        Reads a PDF file and returns a list of chunks with metadata.
        Each chunk is a dictionary containing:
        - doc_id: the base filename
        - chunk_id: unique id for the chunk
        - text: the text content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = os.path.basename(file_path)
        chunks = []

        # Resource limits (PDF = untrusted input). A skipped PDF never aborts the whole build. Note: a
        # hung native parser cannot be killed from Python, so there is deliberately no timeout here.
        size = os.path.getsize(file_path)
        if size > MAX_PDF_BYTES:
            print(f"  [Skip] {doc_id}: {size / 1e6:.0f} MB exceeds the {MAX_PDF_BYTES / 1e6:.0f} MB limit.")
            return []

        try:
            doc = fitz.open(file_path)
            if len(doc) > MAX_PDF_PAGES:
                print(f"  [Skip] {doc_id}: {len(doc)} pages exceeds the {MAX_PDF_PAGES}-page limit.")
                doc.close()
                return []
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text("text") + "\n"

            full_text = _fix_ligatures(full_text)  # ﬁ/ﬂ → fi/fl before anything downstream

            # Drop low-value tail sections (References/Bibliography/Acknowledgements and
            # anything after them) before chunking, so they enter neither the graph nor the
            # vector store. Logged so an unexpected cut is visible instead of silent.
            before = len(full_text)
            full_text = self._filter_sections(full_text)
            if len(full_text) < before:
                pct = 100 * (1 - len(full_text) / before)
                print(f"  [Filter] {doc_id}: dibuang {pct:.0f}% "
                      f"({before - len(full_text):,} char refs/appendix)")

            # Simple chunking by characters
            # In a more advanced setup, we could chunk by sentences or paragraphs.
            start = 0
            chunk_index = 0
            while start < len(full_text):
                end = start + self.chunk_size
                chunk_text = full_text[start:end]
                
                # Try to end at a natural boundary (newline or period)
                if end < len(full_text):
                    last_newline = chunk_text.rfind('\n')
                    last_period = chunk_text.rfind('. ')
                    boundary = max(last_newline, last_period)
                    
                    if boundary > self.chunk_size // 2:
                        end = start + boundary + 1
                        chunk_text = full_text[start:end]

                if chunk_text.strip():
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                        "text": chunk_text.strip()
                    })
                    chunk_index += 1
                
                start = end - self.chunk_overlap

            doc.close()
            return chunks

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def process_directory(self, dir_path: str) -> List[Dict[str, str]]:
        """
        Process all PDFs in a directory.
        """
        all_chunks = []
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(dir_path, filename)
                print(f"Processing {filename}...")
                chunks = self.process_pdf(file_path)
                all_chunks.extend(chunks)
        return all_chunks

if __name__ == "__main__":
    # Test
    processor = DocumentProcessor()
    # chunks = processor.process_directory("../../data/raw")
    # print(f"Extracted {len(chunks)} chunks.")
