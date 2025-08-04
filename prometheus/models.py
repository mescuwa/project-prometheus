from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime  # Add this for timestamp field
from enum import Enum

class CodeChunk(BaseModel):
    chunk_id: str  # SHA-256 hash of canonical_content
    file_path: str
    start_line: int
    end_line: int
    original_content: str  # The exact source code as extracted
    canonical_content: str  # Formatted content used for ID generation
    chunk_type: str  # e.g., 'function', 'class', 'method', 'import_statement_group'
    name: Optional[str] = None  # Name of the function, class, etc.
    embedding_model_name: Optional[str] = None  # New field to track embedding model 
    
    # --- NEW FIELDS FOR VERSIONING ---
    epoch_tag: Optional[int] = None # Alephron's current epoch when indexed
    git_commit_hash: Optional[str] = None # Optional: if project is a git repo
    indexed_at_timestamp: Optional[datetime] = None # Timestamp of indexing
    # --- END NEW FIELDS --- 

class VerificationFailureDetails(BaseModel):
    """Structured details about why Truthkeeper verification failed."""

    type: str
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_offset: Optional[int] = None
    linter_output: Optional[Any] = None  # Can be list of dicts or string
    test_output: Optional[str] = None
    import_error_output: Optional[str] = None
    diff_content: Optional[str] = None


class SymbolReference(BaseModel):
    """Explicit reference to a Python symbol to be (re)created or modified."""

    qname: str = Field(..., description="Qualified name, e.g., path/to/file.py:MyClass.my_method")
    symbol_type: str = Field(..., description="'class', 'function', or 'method'")
    file_path: str
    name: str


class Step(BaseModel):
    id: int
    action: str
    details: Optional[str] = None
    relevant_file_paths: List[str]
    target_file: Optional[str] = None
    target_chunk_id: Optional[str] = None

    # --- NEW FIELD ---
    # Structured dictionary for domain-specific parameters (e.g., docking box coords)
    simulation_parameters: Optional[Dict[str, Any]] = None
    suggested_llm_model: Optional[str] = None
    suggested_temperature: Optional[float] = None
    confidence_score: float = 1.0
    estimated_complexity: int = 1
    allow_refinement: bool = True
    is_fix_for_previous_step: bool = False
    target_symbol: Optional[SymbolReference] = None


class Plan(BaseModel):
    plan_version: str = "1.0"
    steps: List[Step]


class FailureContext(BaseModel):
    failed_step_action: str
    failed_step_details_from_plan: Dict[str, Any]
    final_truthkeeper_verdict: str
    final_verification_details: Optional[VerificationFailureDetails] = None
    final_modifier_result_summary: Optional[Dict[str, Any]] = None
    timestamp: datetime


# ---------------------------------------------------------------------------
# Enums and lightweight config stubs for compatibility
# ---------------------------------------------------------------------------


class PatchMode(str, Enum):
    UNIFIED_DIFF = "UNIFIED_DIFF"
    FULL_CONTENT = "FULL_CONTENT"


# Minimal stub of AppConfig with common fields referenced by code. This is *not*
# a full representation of the user-facing TOML schema but suffices for runtime
# attribute access used in the current Python modules.


class _LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "alephron.log"


class _LlmConfig(BaseModel):
    default_planner_model: str = "gemini-2.5-pro"
    default_modifier_model: str = "gemini-2.5-pro"
    default_temperature: Optional[float] = None


class _CoreSettings(BaseModel):
    project_goal: str = ""
    allowed_dirs: List[str] = ["."]


class _SandboxConfig(BaseModel):
    enable_sandboxing: bool = False


class AppConfig(BaseModel):
    llm: _LlmConfig = _LlmConfig()
    logging: _LoggingConfig = _LoggingConfig()
    project_core_settings: _CoreSettings = _CoreSettings()
    sandbox: _SandboxConfig = _SandboxConfig()

    class Config:
        arbitrary_types_allowed = True 