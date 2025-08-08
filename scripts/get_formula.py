# scripts/get_formula.py
import sys
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

def get_formula_from_smiles(smiles: str) -> str:
    """Calculates the molecular formula from a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        
        # Add explicit hydrogens to get the correct atom count
        mol_with_hs = Chem.AddHs(mol)
        
        # Calculate the formula
        formula = CalcMolFormula(mol_with_hs)
        return formula
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/get_formula.py '<SMILES_STRING>'")
    else:
        smiles_input = sys.argv[1]
        formula_output = get_formula_from_smiles(smiles_input)
        print(f"SMILES: {smiles_input}")
        print(f"Formula: {formula_output}")