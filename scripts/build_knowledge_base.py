# scripts/build_knowledge_base.py
import sys
import logging
from pathlib import Path
import hashlib

import tomli as toml  # For parsing TOML configuration files

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- Imports ---
import lancedb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
    logger.info("Chunking papers into paragraphs…")

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

    # 0. Load Configuration
    logger.info("Loading configuration from config.toml…")
    config_path = project_root / "config.toml"
    try:
        with config_path.open("rb") as f:
            kb_config = toml.load(f)["knowledge_base"]
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Failed to load [knowledge_base] config: {e}", exc_info=True)
        return

    literature_dir = project_root / kb_config["literature_dir"]
    db_path = project_root / kb_config["db_path"]

    # 1. Parse PDFs
    papers = parse_pdfs(literature_dir)
    if not papers:
        return

    # 2. Chunk Papers
    knowledge_chunks = chunk_papers(papers)
    if not knowledge_chunks:
        logger.error("No knowledge chunks were generated. Halting process.")
        return

    # 3. Embed Chunks
    logger.info(f"Loading embedding model: {kb_config['embedding_model_name']}")
    model = SentenceTransformer(kb_config['embedding_model_name'])

    logger.info("Generating vector embeddings for all knowledge chunks…")
    contents_to_embed = [chunk['content'] for chunk in knowledge_chunks]

    # The model.encode() method is highly optimized and will show its own progress bar
    vectors = model.encode(contents_to_embed, show_progress_bar=True)

    # Add the generated vector to each chunk dictionary
    for i, chunk in enumerate(knowledge_chunks):
        chunk['vector'] = vectors[i]

    logger.info("Embeddings generated successfully.")

    # 4. Store in LanceDB
    logger.info(f"Connecting to LanceDB at: {db_path}")
    db = lancedb.connect(db_path)

    # Check if the table already exists and drop it for a fresh start
    if kb_config["table_name"] in db.table_names():
        logger.warning(
            f"Table '{kb_config['table_name']}' already exists. Dropping it for a fresh build.")
        db.drop_table(kb_config["table_name"])

    logger.info(f"Creating new LanceDB table: '{kb_config['table_name']}'")
    db.create_table(kb_config["table_name"], data=knowledge_chunks)

    logger.info(f"Successfully created and populated the table with {len(knowledge_chunks)} chunks.")
    logger.info("--- Knowledge Base Construction Complete ---")


if __name__ == "__main__":
    main()
