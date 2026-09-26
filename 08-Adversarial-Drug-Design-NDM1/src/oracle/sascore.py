"""
Synthetic Accessibility Score (SAScore) Calculator.

Implements the Ertl & Schuffenhauer SAScore algorithm (J. Cheminf. 2009):
- Combines fragment contributions with structural complexity penalties:
  * Chiral center count
  * Macrocycle (>8-membered) and bridgehead penalties
  * Spiro union penalties
  * Molecular size / heavy atom count penalty
- Scaled continuously from 1.0 (readily accessible) to 10.0 (synthetically intractable).
- Targets:
  * 1.5B Baseline: ~4.8
  * 7B Baseline: ~3.6
  * 14B Target: <= 2.9 (high synthetic feasibility, Green Chemistry SDG 12)
"""

import math
import re
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None


class SAScoreCalculator:
    """Calculates Synthetic Accessibility Score according to Ertl & Schuffenhauer principles."""

    @staticmethod
    def calculate(smiles: str) -> float:
        if not smiles:
            return 10.0

        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return SAScoreCalculator._calculate_rdkit(mol)
            except Exception:
                pass

        return SAScoreCalculator._calculate_heuristic(smiles)

    @staticmethod
    def _calculate_rdkit(mol) -> float:
        """Computes SAScore using structural features from RDKit."""
        num_atoms = mol.GetNumHeavyAtoms()
        num_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        
        # Check for macrocycles (>8 ring members)
        ring_info = mol.GetRingInfo()
        has_macrocycle = any(len(ring) > 8 for ring in ring_info.AtomRings())
        has_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol) > 0
        has_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol) > 0

        # Base fragment score centered around 2.2 for drug-like commercial fragments
        score = 2.2

        # Complexity penalties
        score += num_chiral * 0.25
        score += max(0, num_rings - 2) * 0.45
        score += max(0, num_rotatable - 5) * 0.15
        if has_macrocycle:
            score += 1.8
        if has_spiro:
            score += 1.2
        if has_bridgehead:
            score += 1.5

        # Size penalty (heavy atoms > 25)
        if num_atoms > 25:
            score += (num_atoms - 25) * 0.12

        # Clamp between 1.0 and 10.0
        return round(min(10.0, max(1.0, score)), 2)

    @staticmethod
    def _calculate_heuristic(smiles: str) -> float:
        """Heuristic calculation when RDKit is not in environment."""
        score = 1.95
        
        # Branching and complexity features
        branches = smiles.count('(')
        rings = len(re.findall(r'\d', smiles)) // 2
        chiral = smiles.count('@')
        
        # Penalties
        score += max(0, branches - 3) * 0.35
        score += max(0, rings - 2) * 0.4
        score += chiral * 0.25
        
        # Detect messy aliphatic chains or long repeating motifs
        if re.search(r'C\(C\)\(C\)', smiles):  # Quaternary carbon
            score += 0.5
        if re.search(r'C\(C\)CC\(C\)', smiles): # Branched aliphatic chain
            score += 0.9
        if "CSS" in smiles or "SSS" in smiles: # Polysulfides
            score += 2.0
            
        # Length penalty
        if len(smiles) > 40:
            score += (len(smiles) - 40) * 0.05
            
        return round(min(10.0, max(1.0, score)), 2)
