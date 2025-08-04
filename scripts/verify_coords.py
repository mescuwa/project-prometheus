# scripts/verify_coords.py
from pathlib import Path
import MDAnalysis as mda

# --- Configuration ---
# Make sure your PDB file is in the right place
pdb_file = Path(__file__).resolve().parents[1] / "data" / "pdb" / "1m17.pdb"
ligand_residue_name = "AQ4"

# --- Main Script ---
print(f"Loading PDB file: {pdb_file}")

if not pdb_file.exists():
    print(f"ERROR: PDB file not found. Please make sure it's at {pdb_file}")
    exit()

# Load the entire PDB file into an MDAnalysis "Universe"
universe = mda.Universe(str(pdb_file))

# Create a selection for all atoms belonging to the ligand "AQ4"
selection_string = f"resname {ligand_residue_name}"
ligand_atoms = universe.select_atoms(selection_string)

if len(ligand_atoms) == 0:
    print(f"ERROR: Could not find any atoms for residue name '{ligand_residue_name}'.")
    print("Check the PDB file to ensure the ligand has the correct name.")
    exit()

# Calculate the center of mass for the selection
center_of_mass_coords = ligand_atoms.center_of_mass()

# Print the results
print("-" * 30)
print(f"Found {len(ligand_atoms)} atoms for ligand '{ligand_residue_name}'.")
print(f"Verified Center of Mass (X, Y, Z): {center_of_mass_coords}")
print("-" * 30)

# Compare to our mission briefing
expected_coords = [21.857, 0.260, 52.761]
print(f"Expected coordinates: {expected_coords}")

# Check if the calculated values are very close to the expected ones
import numpy as np
if np.allclose(center_of_mass_coords, expected_coords, atol=0.001):
    print("✅ SUCCESS: The calculated coordinates match our mission briefing.")
else:
    print("❌ FAILURE: The calculated coordinates DO NOT match our mission briefing.") 