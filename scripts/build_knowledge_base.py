# scripts/build_knowledge_base.py
import sys
import logging
from pathlib import Path
import hashlib

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- Imports ---
import lancedb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Configuration ---
LITERATURE_DIR = project_root / "data" / "literature"
DB_PATH = project_root / "data" / "knowledge_base.lancedb"
TABLE_NAME = "erlotinib_research"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # A fast, effective model

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_pdfs(directory: Path) -> list[dict]:
    """Finds all PDFs in a directory and extracts their text content."""
    all_papers = []
    logger.info(f"Scanning for PDF files in: {directory}")

    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in the directory. Please add research papers.")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files to process.")

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            full_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            all_papers.append({
                "source": pdf_path.name,
                "content": full_text
            })
            logger.info(f"Successfully parsed: {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to parse {pdf_path.name}: {e}", exc_info=True)

    return all_papers


def chunk_papers(papers: list[dict]) -> list[dict]:
    """Chunks the text of each paper into paragraphs."""
    all_chunks = []
    logger.info("Chunking papers into paragraphs...")

    for paper in papers:
        # Split by double newline, a common paragraph separator
        paragraphs = paper['content'].split('\n\n')
        for para in paragraphs:
            clean_para = para.strip()
            # Filter out very short, likely useless chunks (e.g., page numbers, headers)
            if len(clean_para) > 100:
                chunk_id = hashlib.sha256(clean_para.encode()).hexdigest()
                all_chunks.append({
                    "id": chunk_id,
                    "source": paper['source'],
                    "content": clean_para
                })

    logger.info(f"Generated {len(all_chunks)} knowledge chunks from {len(papers)} papers.")
    return all_chunks


def main():
    """The main function to build our scientific knowledge base."""
    logger.info("--- Starting Knowledge Base Construction ---")

    # 1. Parse PDFs
    papers = parse_pdfs(LITERATURE_DIR)
    if not papers:
        return

    # 2. Chunk Papers
    knowledge_chunks = chunk_papers(papers)
    if not knowledge_chunks:
        logger.error("No knowledge chunks were generated. Halting process.")
        return

    # 3. Embed Chunks
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    logger.info("Generating vector embeddings for all knowledge chunks...")
    # We get the content of each chunk to feed to the model
    contents_to_embed = [chunk['content'] for chunk in knowledge_chunks]

    # The model.encode() method is highly optimized and will show its own progress bar
    vectors = model.encode(contents_to_embed, show_progress_bar=True)

    # Add the generated vector to each chunk dictionary
    for i, chunk in enumerate(knowledge_chunks):
        chunk['vector'] = vectors[i]

    logger.info("Embeddings generated successfully.")

    # 4. Store in LanceDB
    logger.info(f"Connecting to LanceDB at: {DB_PATH}")
    db = lancedb.connect(DB_PATH)

    # Check if the table already exists and drop it for a fresh start
    if TABLE_NAME in db.table_names():
        logger.warning(f"Table '{TABLE_NAME}' already exists. Dropping it for a fresh build.")
        db.drop_table(TABLE_NAME)

    logger.info(f"Creating new LanceDB table: '{TABLE_NAME}'")
    # We pass the first chunk with its vector to define the table schema
    db.create_table(TABLE_NAME, data=knowledge_chunks)

    logger.info(f"Successfully created and populated the table with {len(knowledge_chunks)} chunks.")
    logger.info("--- Knowledge Base Construction Complete ---")


if __name__ == "__main__":
    main() 