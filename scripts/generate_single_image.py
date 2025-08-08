# scripts/generate_single_image.py
"""
A simple, standalone utility to generate a 2D image for a single molecule.
This is a safe way to regenerate champion images without re-running an entire mission.
"""
import sys
from pathlib import Path
import argparse

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- Imports ---
from prometheus.utils import generate_molecule_image

def main():
    parser = argparse.ArgumentParser(description="Generate a 2D molecule image with its formula.")
    parser.add_argument("smiles", help="The SMILES string of the molecule.")
    parser.add_argument("output_path", help="The full path for the output PNG image.")
    parser.add_argument("--title", help="The title to display on the image (e.g., 'Cycle 3 | CS: 9.661').", default="Molecule")
    args = parser.parse_args()

    print(f"Generating image for: {args.smiles}")
    print(f"Saving to: {args.output_path}")

    success = generate_molecule_image(
        smiles=args.smiles,
        output_path=Path(args.output_path),
        molecule_name=args.title
    )

    if success:
        print("✅ Image generated successfully!")
    else:
        print("❌ Failed to generate image.")

if __name__ == "__main__":
    main()
