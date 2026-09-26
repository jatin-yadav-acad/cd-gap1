"""
NDM-1 (PDB: 3SPU) Target Definition and Active Site Architecture.

New Delhi Metallo-beta-lactamase-1 (NDM-1) is a B1 metallo-beta-lactamase.
Active site features:
- Invariant catalytic binuclear zinc core:
  * Zn1 coordinated by His120, His122, His189
  * Zn2 coordinated by Asp124, Cys208, His250
  * Catalytic bridging hydroxide / water ion between Zn1 and Zn2
- Resistance hotspots on flexible loops (L3 and L10) within 5 Angstroms of active site:
  * Met67 (L3 loop): Modulates roof flexibility and steric tolerance.
  * Lys211 (L10 loop): Electrostatic interaction with ligand carboxylates.
  * Asn220 (L10 loop): Key hydrogen bonding anchor near Zn2.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Standard 20 amino acids
AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]

# Clinical resistance hotspots within 5A of the NDM-1 pocket
HOTSPOT_RESIDUES = {
    67: 'M',   # Met67 (L3 loop)
    211: 'K',  # Lys211 (L10 loop)
    220: 'N'   # Asn220 (L10 loop)
}

# Invariant catalytic residues coordinating the binuclear zinc cluster
INVARIANT_CATALYTIC_RESIDUES = {
    'Zn1': [(120, 'H'), (122, 'H'), (189, 'H')],
    'Zn2': [(124, 'D'), (208, 'C'), (250, 'H')]
}

# Full mature sequence of NDM-1 (PDB: 3SPU chain A, residues 29-293 numbered standardly)
NDM1_WT_SEQUENCE = (
    "MELPNIMHPVAKLSTALAAALMLSGCMPGEIRPTIGQQMETGDQRFGDLVFRQLAPNVWQHTSYLDMPGFG"
    "AVASNGLIVRDGGRVLVVDTAWTDDQTAQILNWIKQEINLPVALAVVTHAHQDKMGGMDALHAAGIATYAN"
    "ALSNQLAPQEGMVAAQHSLTFAANGWVEPATAPNFGPLKVFYPGPGHTSDNITVGIDGTDIAFGGCLIKDS"
    "KAKSLGNLGDADTEHYAASARAFGAAFPKASMIVMSHSAPDSRAAITHTARMADKLR"
)


@dataclass(frozen=True)
class Mutation:
    position: int
    wt_aa: str
    mut_aa: str
    
    @property
    def code(self) -> str:
        return f"{self.wt_aa}{self.position}{self.mut_aa}"

    def __repr__(self) -> str:
        return self.code


@dataclass
class PocketState:
    """Represents the mutational status of the NDM-1 active site pocket."""
    mutations: Dict[int, str] = field(default_factory=dict)
    
    def get_residue(self, position: int) -> str:
        if position in self.mutations:
            return self.mutations[position]
        return HOTSPOT_RESIDUES.get(position, '?')
    
    def apply_mutation(self, mut: Mutation) -> 'PocketState':
        new_muts = dict(self.mutations)
        new_muts[mut.position] = mut.mut_aa
        return PocketState(mutations=new_muts)
    
    def is_wildtype(self) -> bool:
        return len(self.mutations) == 0
    
    @property
    def label(self) -> str:
        if not self.mutations:
            return "WT (PDB:3SPU)"
        return "_".join(f"{HOTSPOT_RESIDUES[pos]}{pos}{aa}" for pos, aa in sorted(self.mutations.items()))
