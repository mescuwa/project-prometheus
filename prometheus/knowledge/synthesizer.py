# alephron/acml/vector_store.py
import asyncio
import lancedb
import pyarrow as pa
import numpy as np
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import logging
import hashlib  # For computing EMPTY_LIST_HASH without external merkle module

from .models import CodeChunk
from .embedder import get_embedding_dimension, get_embedding_model_name_loaded

# Hash of an empty list (merkle.py dependency removed)
EMPTY_LIST_HASH = hashlib.sha256(b"{}").hexdigest()

logger = logging.getLogger(__name__)

# class name renamed
class KnowledgeSynthesizer:
    def __init__(self, base_path: Path, project_name: str = "default_project"):
        self.base_path = base_path
        self.project_name = project_name
        self.db_path = self.base_path / f"{self.project_name}_acml_meta.sqlite"
        self.lance_db_path = self.base_path / f"{self.project_name}_lancedb"
        self.table_name = "code_chunks"
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # LanceDB connection and table
        self.db = None
        self.table = None
        self.index = None  # For backward compatibility (used by PlannerAgent)
        
        # In-memory caches for mapping
        self.chunk_id_to_lance_id_map: Dict[str, int] = {}
        self.lance_id_to_chunk_id_map: Dict[int, str] = {}
        self.next_lance_id: int = 0

        self._init_sqlite()
        self._init_lancedb()
        self._load_mappings()

    def _get_db_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_sqlite(self):
        """Initialize SQLite database for metadata storage."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chunk_metadata (
                        chunk_id TEXT PRIMARY KEY,
                        lance_id INTEGER UNIQUE,
                        file_path TEXT NOT NULL,
                        start_line INTEGER NOT NULL,
                        end_line INTEGER NOT NULL,
                        chunk_type TEXT NOT NULL,
                        name TEXT,
                        embedding_model_name TEXT,
                        epoch_tag INTEGER,          -- Store Alephron epoch
                        git_commit_hash TEXT,       -- Store git commit hash
                        indexed_at_timestamp TEXT   -- Store ISO format timestamp
                    )
                """)
                
                # NEW: Symbol Registry Table for Persistent Accomplishment Memory
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS implemented_symbols (
                        symbol_qname TEXT PRIMARY KEY,        -- e.g., src/utils.py:MyClass.my_method
                        file_path TEXT NOT NULL,              -- e.g., src/utils.py
                        symbol_name TEXT NOT NULL,            -- e.g., my_method
                        symbol_type TEXT NOT NULL,            -- 'class', 'function', 'method'
                        parent_symbol_qname TEXT,             -- QName of parent class if method, else NULL
                        epoch_indexed INTEGER,                -- Alephron epoch when this symbol was last confirmed
                        step_id_indexed TEXT,                 -- Manifest step ID that created/last verified it
                        git_hash_indexed TEXT,                -- Git hash when last confirmed
                        indexed_at TEXT,                      -- ISO Timestamp
                        acml_chunk_id TEXT,                   -- Optional: FK to chunk_metadata if it maps 1:1
                        FOREIGN KEY (acml_chunk_id) REFERENCES chunk_metadata(chunk_id)
                    )
                """)
                
                # Create indices for efficient queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_file_path ON implemented_symbols(file_path)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_type ON implemented_symbols(symbol_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_epoch ON implemented_symbols(epoch_indexed)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_step_id ON implemented_symbols(step_id_indexed)")
                
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"SQLite initialization error: {e}", exc_info=True)
            raise

    def _init_lancedb(self):
        """Initialize LanceDB database and table."""
        try:
            logger.info(f"Initializing LanceDB at {self.lance_db_path}...")
            self.db = lancedb.connect(str(self.lance_db_path))
            
            # Check if table exists
            if self.table_name in self.db.table_names():
                self.table = self.db.open_table(self.table_name)
                # Create a mock index object for backward compatibility with PlannerAgent
                self.index = MockIndex(self.table.count_rows())
                logger.info(f"Opened existing LanceDB table '{self.table_name}' with {self.table.count_rows()} chunks.")
            else:
                # Table doesn't exist yet, will be created when first chunks are added
                self.table = None
                self.index = MockIndex(0)
                logger.info(f"LanceDB table '{self.table_name}' will be created on first chunk addition.")
                
        except Exception as e:
            logger.error(f"LanceDB initialization error: {e}", exc_info=True)
            raise

    def _load_mappings(self):
        """Load chunk ID mappings from SQLite."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_id, lance_id FROM chunk_metadata WHERE lance_id IS NOT NULL")
                max_lance_id_so_far = -1
                for row in cursor.fetchall():
                    chunk_id, lance_id = row[0], row[1]
                    self.chunk_id_to_lance_id_map[chunk_id] = lance_id
                    self.lance_id_to_chunk_id_map[lance_id] = chunk_id
                    if lance_id > max_lance_id_so_far:
                        max_lance_id_so_far = lance_id
                self.next_lance_id = max_lance_id_so_far + 1
                logger.info(f"Loaded {len(self.chunk_id_to_lance_id_map)} chunk mappings from SQLite. Next Lance ID: {self.next_lance_id}")
        except sqlite3.Error as e:
            logger.error(f"Error loading mappings from SQLite: {e}", exc_info=True)

    def _create_table_if_needed(self, embedding_dimension: int):
        """Create LanceDB table with proper schema if it doesn't exist."""
        if self.table is not None:
            return
            
        logger.info(f"Creating LanceDB table '{self.table_name}' with embedding dimension {embedding_dimension}...")
        
        # LanceDB prefers simple data format for table creation
        # We'll create the table with the first batch of data instead of empty schema
        self.table = None  # Will be created when data is added
        self.index = MockIndex(0)
        logger.info(f"LanceDB table '{self.table_name}' will be created with first data batch.")

    def add_chunks(self, chunks_with_embeddings: List[Tuple[CodeChunk, np.ndarray]]):
        """Add chunks with embeddings to the vector store."""
        if not chunks_with_embeddings:
            return

        # Prepare data for insertion
        data_to_add = []
        metadata_to_add_list = []
        current_model_name = get_embedding_model_name_loaded()

        for chunk, embedding in chunks_with_embeddings:
            if chunk.chunk_id in self.chunk_id_to_lance_id_map:
                logger.debug(f"Chunk ID {chunk.chunk_id} already in store. Skipping.")
                continue
            if embedding is None:
                logger.warning(f"Chunk ID {chunk.chunk_id} has no embedding. Skipping.")
                continue

            current_lance_id = self.next_lance_id
            
            # Prepare LanceDB row - simplified format
            data_to_add.append({
                "lance_id": current_lance_id,
                "chunk_id": chunk.chunk_id,
                "vector": embedding.astype(np.float32),  # Keep as numpy array, not list
                "code": chunk.canonical_content,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name or "",
                "embedding_model_name": current_model_name or ""
            })

            # Prepare SQLite metadata
            metadata_to_add_list.append((
                chunk.chunk_id, current_lance_id, chunk.file_path,
                chunk.start_line, chunk.end_line, chunk.chunk_type,
                chunk.name, current_model_name,
                chunk.epoch_tag, 
                chunk.git_commit_hash,
                chunk.indexed_at_timestamp.isoformat() if chunk.indexed_at_timestamp else None
            ))
            
            # Update in-memory mappings
            self.chunk_id_to_lance_id_map[chunk.chunk_id] = current_lance_id
            self.lance_id_to_chunk_id_map[current_lance_id] = chunk.chunk_id
            self.next_lance_id += 1

        if data_to_add:
            try:
                # Create table if it doesn't exist
                if self.table is None:
                    logger.info(f"Creating LanceDB table '{self.table_name}' with {len(data_to_add)} initial chunks...")
                    self.table = self.db.create_table(self.table_name, data_to_add)
                    logger.info(f"Created LanceDB table '{self.table_name}' successfully.")
                else:
                    # Add to existing table
                    self.table.add(data_to_add)
                    logger.info(f"Added {len(data_to_add)} new chunks to LanceDB table.")
                
                # Update mock index
                self.index = MockIndex(self.table.count_rows())
                
            except Exception as e:
                logger.error(f"Error adding chunks to LanceDB: {e}", exc_info=True)
                return

            try:
                # Add metadata to SQLite
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.executemany("""
                        INSERT INTO chunk_metadata (chunk_id, lance_id, file_path, start_line, end_line, chunk_type, name, embedding_model_name, epoch_tag, git_commit_hash, indexed_at_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, metadata_to_add_list)
                    conn.commit()
                logger.info(f"Added {len(metadata_to_add_list)} new metadata entries to SQLite.")
            except sqlite3.Error as e:
                logger.error(f"Error adding metadata to SQLite: {e}", exc_info=True)
                return

    def search_similar_chunks(self, query_embedding: np.ndarray, k: int = 5, 
                            max_epoch_tag: Optional[int] = None, 
                            at_git_hash: Optional[str] = None) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Search for similar chunks using LanceDB vector search with optional version filtering.
        
        Args:
            query_embedding: The query vector for semantic search
            k: Number of top results to return
            max_epoch_tag: If provided, only return chunks with epoch_tag <= max_epoch_tag 
                          (or None epoch_tag for pre-epoch data)
            at_git_hash: If provided, only return chunks with matching git_commit_hash
        
        Returns:
            List of tuples (chunk_id, distance, metadata_dict) sorted by similarity
        """
        results: List[Tuple[str, float, Optional[Dict]]] = []
        
        if self.table is None or self.table.count_rows() == 0:
            logger.warning("LanceDB table not initialized or empty. Cannot search.")
            return results
        if query_embedding is None:
            logger.warning("Query embedding is None. Cannot search.")
            return results

        try:
            # Ensure query_embedding is float32
            query_vector = query_embedding.astype(np.float32).tolist()
            
            # Request more candidates than needed since we'll filter by version
            search_multiplier = 3 if (max_epoch_tag is not None or at_git_hash is not None) else 1
            actual_k = min(k * search_multiplier, self.table.count_rows())
            
            if actual_k == 0:
                return []

            # Perform vector search
            search_results = self.table.search(query_vector).limit(actual_k).to_list()
            
            # Convert results and apply version filtering
            filtered_results = []
            
            for result in search_results:
                chunk_id = result["chunk_id"]
                # LanceDB returns distance, smaller is better (similar to FAISS L2)
                distance = float(result["_distance"])
                
                # Get metadata from SQLite for version filtering
                metadata = self.get_chunk_metadata_by_id(chunk_id)
                if not metadata:
                    continue
                
                # Apply version filters
                passes_filters = True
                
                # Epoch filter
                if max_epoch_tag is not None:
                    chunk_epoch = metadata.get('epoch_tag')
                    if chunk_epoch is not None and chunk_epoch > max_epoch_tag:
                        passes_filters = False
                
                # Git hash filter  
                if passes_filters and at_git_hash is not None:
                    chunk_git_hash = metadata.get('git_commit_hash')
                    if chunk_git_hash != at_git_hash:
                        passes_filters = False
                
                if passes_filters:
                    filtered_results.append((chunk_id, distance, metadata))
                
                # Stop early if we have enough filtered results
                if len(filtered_results) >= k:
                    break

            # Sort by distance (smaller is better) and return top k
            filtered_results.sort(key=lambda x: x[1])
            return filtered_results[:k]

        except Exception as e:
            logger.error(f"Error during LanceDB search: {e}", exc_info=True)
            return []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk and its details by its unique chunk_id from LanceDB."""
        if self.table is None:
            logger.warning("LanceDB table not initialized. Cannot get chunk by ID.")
            return None
        
        try:
            # LanceDB's SQL-like query to find the chunk by its chunk_id
            # The fields selected here must match what ModifierAgent expects:
            # canonical_content (as 'code'), file_path, start_line, end_line
            result = self.table.search() \
                .where(f"chunk_id = '{chunk_id}'") \
                .select(["code", "file_path", "start_line", "end_line"]) \
                .limit(1) \
                .to_list()

            if result and len(result) > 0:
                chunk_data = result[0]
                # Rename 'code' to 'canonical_content' for ModifierAgent's expectation
                return {
                    "canonical_content": chunk_data.get("code"),
                    "file_path": chunk_data.get("file_path"),
                    "start_line": chunk_data.get("start_line"),
                    "end_line": chunk_data.get("end_line")
                }
            else:
                logger.warning(f"Chunk with chunk_id '{chunk_id}' not found in LanceDB table.")
                return None
        except Exception as e:
            logger.error(f"Error retrieving chunk '{chunk_id}' from LanceDB: {e}", exc_info=True)
            return None

    def get_chunk_metadata_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Public method to get metadata, establishing its own DB connection."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                return self.get_chunk_metadata_by_id_internal(chunk_id, cursor)
        except sqlite3.Error as e:
            logger.error(f"SQLite error fetching metadata for {chunk_id}: {e}", exc_info=True)
            return None

    def get_chunk_metadata_by_id_internal(self, chunk_id: str, cursor: sqlite3.Cursor) -> Optional[Dict]:
        """Internal method using an existing cursor."""
        try:
            cursor.execute("SELECT chunk_id, file_path, start_line, end_line, chunk_type, name, embedding_model_name, epoch_tag, git_commit_hash, indexed_at_timestamp FROM chunk_metadata WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "chunk_id": row[0], "file_path": row[1], "start_line": row[2],
                    "end_line": row[3], "chunk_type": row[4], "name": row[5],
                    "embedding_model_name": row[6], "epoch_tag": row[7],
                    "git_commit_hash": row[8], "indexed_at_timestamp": row[9]
                }
            return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving chunk metadata by ID '{chunk_id}': {e}")
            return None

    def save_vector_store_index(self):
        """Save vector store index (for LanceDB, data is persisted automatically)."""
        # LanceDB persists data automatically, but we can log the operation for consistency
        if self.table is not None:
            try:
                count = self.table.count_rows()
                logger.info(f"LanceDB vector store contains {count} chunks (auto-persisted).")
            except Exception as e:
                logger.error(f"Error checking LanceDB table status: {e}", exc_info=True)
        else:
            logger.info("LanceDB table not initialized, nothing to save.")
    
    def get_all_chunk_ids_in_store(self) -> List[str]:
        """Returns a list of all chunk_ids present in the SQLite metadata table."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_id FROM chunk_metadata")
                return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error fetching all chunk IDs from SQLite: {e}", exc_info=True)
            return []

    def get_all_chunks_for_file(self, file_path_str: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific file with versioning metadata."""
        if self.table is None or self.table.count_rows() == 0:
            logger.debug(f"Vector store is empty or not initialized. No chunks for file '{file_path_str}'.")
            return []
        
        try:
            # Query LanceDB table by file_path
            chunks_data = self.table.search().where(f"file_path = '{file_path_str}'").limit(1000).to_list()
            if not chunks_data:
                logger.debug(f"No chunks found for file '{file_path_str}'.")
                return []

            results = []
            for row in chunks_data:
                chunk_id = row['chunk_id']
                
                # Get versioning metadata from SQLite
                metadata = self.get_chunk_metadata_by_id(chunk_id)
                
                chunk_data = {
                    'chunk_id': chunk_id,
                    'canonical_content': row['code'],
                    'file_path': row['file_path'],
                    'start_line': row['start_line'],
                    'end_line': row['end_line'],
                    'chunk_type': row['chunk_type'],
                    'name': row['name']
                }
                
                # Include versioning metadata if available
                if metadata:
                    chunk_data.update({
                        'epoch_tag': metadata.get('epoch_tag'),
                        'git_commit_hash': metadata.get('git_commit_hash'),
                        'indexed_at_timestamp': metadata.get('indexed_at_timestamp'),
                        'embedding_model_name': metadata.get('embedding_model_name')
                    })
                
                results.append(chunk_data)
            
            logger.debug(f"Found {len(results)} chunks for file '{file_path_str}'.")
            return results
        except Exception as e:
            logger.error(f"Error querying chunks for file '{file_path_str}': {e}", exc_info=True)
            return []

    def search_similar_chunks_in_file(self, query_text: str, target_file_path: str, max_results: int = 5,
                                     max_epoch_tag: Optional[int] = None, 
                                     at_git_hash: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks within a specific file with optional version filtering.
        
        Args:
            query_text: The text to search for
            target_file_path: The file path to search within
            max_results: Maximum number of results to return
            max_epoch_tag: If provided, only return chunks with epoch_tag <= max_epoch_tag
            at_git_hash: If provided, only return chunks with matching git_commit_hash
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        if self.table is None or self.table.count_rows() == 0:
            logger.debug(f"Vector store is empty or not initialized. No chunks for similarity search in file '{target_file_path}'.")
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = generate_embedding(query_text)
            if query_embedding is None:
                logger.warning(f"Could not generate embedding for query text: {query_text[:100]}...")
                return []
            
            # Search with file path filter - request more results if version filtering is needed
            search_multiplier = 3 if (max_epoch_tag is not None or at_git_hash is not None) else 1
            search_limit = max_results * search_multiplier
            
            search_results = self.table.search(query_embedding).where(f"file_path = '{target_file_path}'").limit(search_limit).to_list()
            
            if not search_results:
                logger.debug(f"No similar chunks found for query in file '{target_file_path}'.")
                return []

            results = []
            for row in search_results:
                chunk_id = row['chunk_id']
                
                # Get metadata for version filtering  
                metadata = self.get_chunk_metadata_by_id(chunk_id)
                if not metadata:
                    continue
                
                # Apply version filters
                passes_filters = True
                
                # Epoch filter
                if max_epoch_tag is not None:
                    chunk_epoch = metadata.get('epoch_tag')
                    if chunk_epoch is not None and chunk_epoch > max_epoch_tag:
                        passes_filters = False
                
                # Git hash filter
                if passes_filters and at_git_hash is not None:
                    chunk_git_hash = metadata.get('git_commit_hash')
                    if chunk_git_hash != at_git_hash:
                        passes_filters = False
                
                if passes_filters:
                    chunk_data = {
                        'chunk_id': chunk_id,
                        'file_path': row['file_path'],
                        'chunk_type': row['chunk_type'],
                        'name': row['name'],
                        'start_line': row['start_line'],
                        'end_line': row['end_line'],
                        'canonical_content': row['code'],  # Use 'code' field from LanceDB
                        'distance': row.get('_distance', 0.0)  # LanceDB provides distance in search results
                    }
                    results.append(chunk_data)
                
                # Stop early if we have enough filtered results
                if len(results) >= max_results:
                    break
            
            logger.debug(f"Found {len(results)} similar chunks in file '{target_file_path}' for query.")
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar chunks in file '{target_file_path}': {e}", exc_info=True)
            return []

    def delete_chunks_by_file_path(self, file_path_str: str) -> int:
        """
        Deletes all chunks associated with a specific file_path from both
        LanceDB and SQLite metadata.

        Args:
            file_path_str: The relative path of the file whose chunks are to be deleted.

        Returns:
            The number of chunks successfully deleted.
        """
        logger.info(f"ACML Store: Deleting chunks for file: {file_path_str}")
        deleted_count = 0

        # --- Get chunk_ids and lance_ids from SQLite for the given file_path ---
        lance_ids_to_delete_in_lance = []
        chunk_ids_to_delete_from_sqlite = []
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_id, lance_id FROM chunk_metadata WHERE file_path = ?", (file_path_str,))
                rows = cursor.fetchall()
                for row in rows:
                    chunk_ids_to_delete_from_sqlite.append(row[0])
                    if row[1] is not None:  # lance_id might be null if only metadata was added
                        lance_ids_to_delete_in_lance.append(row[1])
                        
                logger.debug(f"ACML Store: Found {len(chunk_ids_to_delete_from_sqlite)} chunks to delete for file {file_path_str}")
                
        except sqlite3.Error as e:
            logger.error(f"ACML Store: Error fetching chunk/lance IDs for deletion from SQLite for {file_path_str}: {e}", exc_info=True)
            return 0  # Abort if we can't reliably get IDs

        # --- Delete from LanceDB ---
        if self.table is not None and chunk_ids_to_delete_from_sqlite:
            try:
                # LanceDB delete uses a SQL-like predicate.
                # Delete by file_path is simpler if that field is reliably in LanceDB.
                if self.table.count_rows() > 0:  # Only if table has rows
                    logger.debug(f"ACML Store: Attempting to delete from LanceDB where file_path = '{file_path_str}'")
                    # Use file_path for deletion - LanceDB will handle the SQL-like predicate
                    self.table.delete(f"file_path = '{file_path_str}'")
                    logger.info(f"ACML Store: LanceDB delete operation completed for file_path = '{file_path_str}'.")
                    
                    # Update mock index after deletion
                    self.index = MockIndex(self.table.count_rows())

            except Exception as e_lance_del:
                logger.error(f"ACML Store: Error deleting chunks from LanceDB for {file_path_str}: {e_lance_del}", exc_info=True)
                # Potentially inconsistent state, but proceed to try SQLite cleanup.

        # --- Delete from SQLite metadata ---
        if chunk_ids_to_delete_from_sqlite:
            try:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    # Delete by chunk_id for safety
                    placeholders = ','.join('?' for _ in chunk_ids_to_delete_from_sqlite)
                    cursor.execute(f"DELETE FROM chunk_metadata WHERE chunk_id IN ({placeholders})", chunk_ids_to_delete_from_sqlite)
                    conn.commit()
                    deleted_count = cursor.rowcount
                    logger.info(f"ACML Store: Deleted {deleted_count} chunk metadata entries from SQLite for {file_path_str}.")
            except sqlite3.Error as e_sqlite_del:
                logger.error(f"ACML Store: Error deleting chunk metadata from SQLite for {file_path_str}: {e_sqlite_del}", exc_info=True)
                return deleted_count  # Return whatever SQLite managed to delete

        # --- Update in-memory mappings ---
        for chunk_id_del in chunk_ids_to_delete_from_sqlite:
            if chunk_id_del in self.chunk_id_to_lance_id_map:
                lance_id_val = self.chunk_id_to_lance_id_map.pop(chunk_id_del)
                if lance_id_val in self.lance_id_to_chunk_id_map:
                    self.lance_id_to_chunk_id_map.pop(lance_id_val)
        
        logger.info(f"ACML Store: Finished deletion process for file {file_path_str}. SQLite deleted: {deleted_count}.")
        return deleted_count

    async def close(self):
        """Cleanup method."""
        pass

    # === SYMBOL REGISTRY METHODS ===
    
    def add_implemented_symbols(self, symbols_data: List[Dict[str, Any]]) -> bool:
        """
        Add or update implemented symbols in the registry.
        
        Args:
            symbols_data: List of dicts with keys: symbol_qname, file_path, symbol_name, 
                         symbol_type, parent_symbol_qname, epoch_indexed, step_id_indexed, 
                         git_hash_indexed, indexed_at, acml_chunk_id
        
        Returns:
            True if successful, False otherwise
        """
        if not symbols_data:
            return True
            
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO implemented_symbols 
                    (symbol_qname, file_path, symbol_name, symbol_type, parent_symbol_qname, 
                     epoch_indexed, step_id_indexed, git_hash_indexed, indexed_at, acml_chunk_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        symbol['symbol_qname'], symbol['file_path'], symbol['symbol_name'],
                        symbol['symbol_type'], symbol.get('parent_symbol_qname'),
                        symbol['epoch_indexed'], symbol['step_id_indexed'],
                        symbol.get('git_hash_indexed'), symbol['indexed_at'],
                        symbol.get('acml_chunk_id')
                    ) for symbol in symbols_data
                ])
                conn.commit()
                logger.info(f"SymbolRegistry: Added/updated {len(symbols_data)} symbols")
                return True
        except sqlite3.Error as e:
            logger.error(f"Error adding symbols to registry: {e}")
            return False
    
    def check_symbol_exists(self, symbol_qname: str) -> bool:
        """Check if a symbol with the given qualified name exists in the registry."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM implemented_symbols WHERE symbol_qname = ?", (symbol_qname,))
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Error checking symbol existence for '{symbol_qname}': {e}")
            return False
    
    def get_symbols_by_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all symbols for a specific file."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol_qname, file_path, symbol_name, symbol_type, parent_symbol_qname,
                           epoch_indexed, step_id_indexed, git_hash_indexed, indexed_at, acml_chunk_id
                    FROM implemented_symbols 
                    WHERE file_path = ?
                    ORDER BY symbol_name
                """, (file_path,))
                
                columns = ['symbol_qname', 'file_path', 'symbol_name', 'symbol_type', 'parent_symbol_qname',
                          'epoch_indexed', 'step_id_indexed', 'git_hash_indexed', 'indexed_at', 'acml_chunk_id']
                
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error getting symbols for file '{file_path}': {e}")
            return []
    
    def get_symbols_by_type(self, symbol_type: str) -> List[Dict[str, Any]]:
        """Get all symbols of a specific type (class, function, method)."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol_qname, file_path, symbol_name, symbol_type, parent_symbol_qname,
                           epoch_indexed, step_id_indexed, git_hash_indexed, indexed_at, acml_chunk_id
                    FROM implemented_symbols 
                    WHERE symbol_type = ?
                    ORDER BY file_path, symbol_name
                """, (symbol_type,))
                
                columns = ['symbol_qname', 'file_path', 'symbol_name', 'symbol_type', 'parent_symbol_qname',
                          'epoch_indexed', 'step_id_indexed', 'git_hash_indexed', 'indexed_at', 'acml_chunk_id']
                
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error getting symbols by type '{symbol_type}': {e}")
            return []

    def remove_symbols_by_file(self, file_path: str) -> bool:
        """Remove all symbols for a specific file (useful when file is deleted)."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM implemented_symbols WHERE file_path = ?", (file_path,))
                deleted_count = cursor.rowcount
                conn.commit()
                logger.info(f"SymbolRegistry: Removed {deleted_count} symbols for file '{file_path}'")
                return True
        except sqlite3.Error as e:
            logger.error(f"Error removing symbols for file '{file_path}': {e}")
            return False

class MockIndex:
    """Mock index object for backward compatibility with PlannerAgent's index checks."""
    def __init__(self, ntotal: int):
        self.ntotal = ntotal
        self.d = get_embedding_dimension() or 384  # Default dimension if not available
    
    def __bool__(self):
        return True  # Always truthy so PlannerAgent thinks index exists 