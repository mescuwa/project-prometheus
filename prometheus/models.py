from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class VerificationFailureDetails(BaseModel):
    """Structured details about why an experiment's validation failed."""

    type: str
    message: str
    test_output: Optional[str] = None
    # Additional contextual fields can be added here as needed.


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

    # Domain-specific parameters (e.g., docking box coordinates)
    simulation_parameters: Optional[Dict[str, Any]] = None

    suggested_llm_model: Optional[str] = None
    suggested_temperature: Optional[float] = None

    confidence_score: float = 1.0
    estimated_complexity: int = 1
    allow_refinement: bool = True
    is_fix_for_previous_step: bool = False
    target_symbol: Optional[SymbolReference] = None


class ExperimentLog(BaseModel):
    """A record of a single experiment performed during the autonomous loop."""
    cycle: int
    smiles: str
    reasoning: Optional[str] = None
    average_binding_affinity: float
    std_dev_binding_affinity: Optional[float] = None

    # Multi-objective property scores
    qed: Optional[float] = None
    sa_score: Optional[float] = None
    logp: Optional[float] = None
    composite_score: Optional[float] = None
    
    # Path to a generated 2D depiction of the molecule (optional)
    image_path: Optional[str] = None

    verdict: str
