# prometheus/utils.py
"""Utility helpers for Project Prometheus."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)


def generate_molecule_image(
    smiles: str,
    output_path: Path,
    molecule_name: str = "Molecule",
    highlight_smarts: str | None = None,
    size: tuple[int, int] = (450, 400),
) -> bool:
    """Generate a 2-D depiction (PNG) of *smiles* at *output_path*."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error("Could not parse SMILES – skipping image generation: %s", smiles)
            return False

        # --- NEW: Automatically calculate and add the formula ---
        try:
            mol_with_hs = Chem.AddHs(mol)
            formula = CalcMolFormula(mol_with_hs)
            # Create a rich legend with the name and the formula
            legend = f"{molecule_name} | {formula}"
        except Exception:
            # Fallback to the original name if formula calculation fails
            legend = molecule_name
        # --- END OF NEW CODE ---

        # Explicitly generate a single, clean 2D conformation.
        AllChem.Compute2DCoords(mol)

        try:
            Chem.Kekulize(mol)
        except Exception:
            pass

        width, height = size
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol, legend=legend)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from cairosvg import svg2png
            svg2png(bytestring=svg.encode(), write_to=str(output_path))
        except ImportError:
            logger.debug("cairosvg not available – falling back to RDKit PNG rendering.")
            img = Draw.MolToImage(mol, size=size, legend=legend)
            img.save(output_path)

        logger.debug("Generated molecule image at %s", output_path)
        return True
    except Exception as exc:
        logger.error("Failed to generate molecule image for %s: %s", smiles, exc, exc_info=True)
        return False


def setup_logging(project_root: Path) -> None:
    """Configures logging to output to both console and a timestamped file."""
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"prometheus_run_{timestamp}.log"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s – %(levelname)s – %(message)s")
    console_handler.setFormatter(console_formatter)
    
    file_handler = RotatingFileHandler(log_filename, maxBytes=5*1024*1024, backupCount=2)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s – %(name)s – %(levelname)s – %(message)s")
    file_handler.setFormatter(file_formatter)
    
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured. Detailed logs will be saved to: {log_filename}")