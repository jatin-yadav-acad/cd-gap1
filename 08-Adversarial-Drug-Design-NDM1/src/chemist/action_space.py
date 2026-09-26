"""
Chemist Agent Action Space: Functional Group Additions, Substitutions, and Extensions.

Implements chemical transformations to re-establish affinity against mutated NDM-1 pockets:
- Thiol linkers (Zn1/Zn2 catalytic coordination)
- Carboxylate arms (Lys211/Asn220 electrostatic and H-bonding)
- Bicyclic rings & rigid scaffolds (L3 loop conformational stability)
- Aromatic/heteroaryl extensions (filling hydrophobic voids, e.g. M67V)
- Bis-zinc invariant chelators (terminal Nash equilibrium attractor)
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass(frozen=True)
class ChemAction:
    action_id: str
    name: str
    category: str
    description: str
    target_site: str
    smiles_transformation: str


# Chemical transformation catalogue
CHEM_ACTIONS: List[ChemAction] = [
    # 1. Thiol Linkers (Direct Zinc Coordination)
    ChemAction(
        action_id="add_thiol_linker_alpha",
        name="Alpha-Thiol Methylation",
        category="zinc_coordination",
        description="Introduces an alpha-methylthiol linker to optimize thiolate coordination between Zn1 and Zn2.",
        target_site="alpha_carbon",
        smiles_transformation="extend_thiol_alpha"
    ),
    ChemAction(
        action_id="add_dithiol_bridge",
        name="Bis-Thiol Chelator Extension",
        category="zinc_coordination",
        description="Appends a vicinal or geminal thiol arm for dual coordination across the Zn1-Zn2 binuclear bridge.",
        target_site="terminal_chain",
        smiles_transformation="dithiol_bridge"
    ),

    # 2. Carboxylate Arms (Engaging Lys211 & Asn220)
    ChemAction(
        action_id="extend_carboxylate_arm",
        name="Beta-Carboxylate Extension",
        category="electrostatic_anchor",
        description="Appends a carboxylate arm to form a salt bridge with Lys211 and H-bonds with Asn220.",
        target_site="c_terminus",
        smiles_transformation="extend_carboxylate"
    ),
    ChemAction(
        action_id="add_dicarboxylate_motif",
        name="Dicarboxylic Acid Clamp",
        category="electrostatic_anchor",
        description="Installs a branched dicarboxylate motif to maintain electrostatic binding even upon K211N/Q mutation.",
        target_site="proline_ring",
        smiles_transformation="dicarboxylate_clamp"
    ),

    # 3. Bicyclic Rings & Rigid Scaffolds
    ChemAction(
        action_id="proline_to_octahydroindole",
        name="Octahydroindole Bicyclic Fusion",
        category="conformational_rigidity",
        description="Fuses a cyclohexane ring onto the proline core to lock conformation against the flexible L3 loop.",
        target_site="core_ring",
        smiles_transformation="fuse_octahydroindole"
    ),
    ChemAction(
        action_id="proline_to_thiazolidine",
        name="Thiazolidine Substitution",
        category="conformational_rigidity",
        description="Substitutes the pyrrolidine ring with thiazolidine-4-carboxylate to enhance metabolic and binding stability.",
        target_site="core_ring",
        smiles_transformation="substitute_thiazolidine"
    ),
    ChemAction(
        action_id="proline_to_tetrahydroisoquinoline",
        name="Tetrahydroisoquinoline (TIC) Extension",
        category="conformational_rigidity",
        description="Replaces pyrrolidine with a TIC scaffold for aromatic pi-stacking with Phe70.",
        target_site="core_ring",
        smiles_transformation="substitute_tic"
    ),

    # 4. Hydrophobic Pocket Fillers (Targeting M67V / M67I voids)
    ChemAction(
        action_id="add_benzyl_substituent",
        name="P1-Benzyl Extension",
        category="hydrophobic_filler",
        description="Appends a benzyl ring into the P1 pocket to fill the hydrophobic cavity opened by M67V.",
        target_site="beta_carbon",
        smiles_transformation="add_benzyl"
    ),
    ChemAction(
        action_id="add_fluorobenzyl_substituent",
        name="4-Fluorobenzyl Shield",
        category="hydrophobic_filler",
        description="Introduces a 4-fluorophenyl group to increase lipophilicity and resist hydrolytic expulsion.",
        target_site="beta_carbon",
        smiles_transformation="add_fluorobenzyl"
    ),
    ChemAction(
        action_id="add_thiophene_linker",
        name="Thiophene Heteroaromatic Ring",
        category="hydrophobic_filler",
        description="Appends a 2-thiophene ring targeting L3 loop residues with optimal aromatic geometry.",
        target_site="linker",
        smiles_transformation="add_thiophene"
    ),

    # 5. Invariant Zinc-Chelating Terminal Scaffold (Nash Equilibrium Attractor)
    ChemAction(
        action_id="install_invariant_bis_zinc_clamp",
        name="Invariant Bis-Zinc Tridentate Clamp",
        category="terminal_equilibrium",
        description="Forms a tridentate thiol-carboxylate-hydroxyl coordination sphere that binds Zn1 and Zn2 simultaneously.",
        target_site="catalytic_core",
        smiles_transformation="install_bis_zinc_clamp"
    )
]
