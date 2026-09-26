"""
Seed chemical fragments and pharmacophore scaffolds for NDM-1 inhibition.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ChemicalScaffold:
    name: str
    smiles: str
    description: str
    zinc_binding_group: str
    canonical_smiles: str


# Seed scaffolds matching the specification: Captopril core & Thienamycin core
SEED_SCAFFOLDS: Dict[str, ChemicalScaffold] = {
    "captopril_core": ChemicalScaffold(
        name="Captopril Core",
        smiles="CC(CS)C(=O)N1CCCC1C(=O)O",
        description="D-captopril scaffold featuring a free terminal thiol for zinc chelation and a proline ring.",
        zinc_binding_group="thiol (-SH)",
        canonical_smiles="CC(CS)C(=O)N1CCCC1C(=O)O"
    ),
    "thienamycin_core": ChemicalScaffold(
        name="Thienamycin Core Fragment",
        smiles="CC(O)C1C2CC(=O)N2C=C1SCC",
        description="Carbapenem core fragment presenting beta-lactam carbonyl and thioether arm.",
        zinc_binding_group="beta-lactam / carboxylate",
        canonical_smiles="CC(O)C1C2CC(=O)N2C=C1SCC"
    ),
    "mercaptocarboxylate_seed": ChemicalScaffold(
        name="Mercaptocarboxylate Dual-Anchor",
        smiles="SCC(C(=O)O)CC1=CC=CC=C1",
        description="Dual-anchor fragment with adjacent thiol and carboxylate for bis-zinc coordination.",
        zinc_binding_group="thiol + carboxylate",
        canonical_smiles="OC(=O)C(CS)Cc1ccccc1"
    )
}
