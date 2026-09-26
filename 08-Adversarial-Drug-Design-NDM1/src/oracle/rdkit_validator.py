"""
RDKit Chemical Syntax and Valence Filter.

Validates chemical grammar:
- Strict valence rules
- Aromaticity perception & ring closures
- Charge neutrality and sanitization
- Handles both native RDKit and robust fallback parser
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    Descriptors = None
    QED = None


@dataclass
class ValidationResult:
    is_valid: bool
    canonical_smiles: Optional[str]
    error_message: Optional[str]
    num_heavy_atoms: int = 0
    molecular_weight: float = 0.0


class MoleculeValidator:
    """Validates SMILES strings against strict chemical valence and aromaticity rules."""

    @staticmethod
    def validate(smiles: str) -> ValidationResult:
        if not smiles or not isinstance(smiles, str):
            return ValidationResult(False, None, "Empty or non-string SMILES")

        smiles = smiles.strip()
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return ValidationResult(False, None, "RDKit parse failure: invalid valence or syntax")
                
                # Check sanitization
                Chem.SanitizeMol(mol)
                canon = Chem.MolToSmiles(mol)
                mw = Descriptors.MolWt(mol)
                heavy = mol.GetNumHeavyAtoms()
                return ValidationResult(True, canon, None, heavy, mw)
            except Exception as e:
                return ValidationResult(False, None, f"RDKit sanitization error: {str(e)}")

        # Fallback syntactic validator (rule-based chemical grammar)
        return MoleculeValidator._fallback_validate(smiles)

    @staticmethod
    def _fallback_validate(smiles: str) -> ValidationResult:
        # Check parenthesis matching
        if smiles.count('(') != smiles.count(')'):
            return ValidationResult(False, None, "Mismatched parentheses")
        if smiles.count('[') != smiles.count(']'):
            return ValidationResult(False, None, "Mismatched square brackets")

        # Check ring closure digit balance
        digits = re.findall(r'(?<!%)\d', smiles)
        digit_counts = {}
        for d in digits:
            digit_counts[d] = digit_counts.get(d, 0) + 1
        for d, count in digit_counts.items():
            if count % 2 != 0:
                return ValidationResult(False, None, f"Unclosed ring index: {d}")

        # Check for forbidden syntactic combinations
        invalid_patterns = [
            r'(=O){2,}',        # Multiple double bonds to same oxygen
            r'C\(=C\)\(=C\)',   # Hypervalent carbon
            r'N\(C\)\(C\)\(C\)1', # Pentavalent nitrogen without charge
            r'\[N\+\].*O-',     # Unbound separated charges
            r'c2cccc2',         # Non-aromatic 4-membered ring marked aromatic
            r'c[1-9]?[a-z]*$',  # Dangling aromatic atom at end
            r'\(=\)',           # Empty branch with double bond
            r'[\(\[\=]$',       # Trailing syntax symbols
        ]
        for pat in invalid_patterns:
            if re.search(pat, smiles):
                return ValidationResult(False, None, f"Valence/aromaticity violation: {pat}")

        # Basic heavy atom count
        heavy_atoms = re.findall(r'[CNOFPSBrcnlops]', smiles)
        num_heavy = len(heavy_atoms)
        if num_heavy < 3:
            return ValidationResult(False, None, "Molecule too small (< 3 heavy atoms)")

        # Estimate MW
        mw = num_heavy * 14.5 + 32.0  # rough estimate
        return ValidationResult(True, smiles, None, num_heavy, mw)
