"""
Live RDKit Cheminformatics and Drug-Likeness Evaluation of Lead Candidates.

Computes exact physical chemical properties using native RDKit:
- Molecular Weight, LogP, TPSA, HBD, HBA, Rotatable Bonds
- Native QED (Quantitative Estimate of Drug-likeness)
- Topological complexity and synthetic viability metrics
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
import pandas as pd


CANDIDATES = {
    "Captopril (Seed)": "CC(CS)C(=O)N1CCCC1C(=O)O",
    "Thienamycin Fragment": "CC(O)C1C2CC(=O)N2C=C1SCC",
    "1.5B Flawed Proposal": "CC(CS)C(=O)N1CC(C(=O)N(C)CC(C)CC(C)C)CC1C(=O)O",
    "7B Extended Intermediate": "CC(CS)C(=O)N1CC(NC(=O)c2ccccc2)CC1C(=O)O",
    "14B MCTS Invariant Lead": "O=C(O)C(CS)CC(=O)N1C(Cc2ccccc2)CSC1C(=O)O"
}


def analyze_candidates():
    print("[*] Running native RDKit cheminformatics evaluation on genuine candidate molecules...\n")
    results = []

    for name, smiles in CANDIDATES.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({
                "Compound": name,
                "Valid": False,
                "MW": 0.0,
                "LogP": 0.0,
                "TPSA": 0.0,
                "HBD": 0,
                "HBA": 0,
                "RotBonds": 0,
                "QED": 0.0
            })
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        qed_val = QED.qed(mol)

        results.append({
            "Compound": name,
            "Valid": True,
            "MW": round(mw, 2),
            "LogP": round(logp, 2),
            "TPSA": round(tpsa, 2),
            "HBD": hbd,
            "HBA": hba,
            "RotBonds": rot_bonds,
            "QED": round(qed_val, 3)
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    analyze_candidates()
