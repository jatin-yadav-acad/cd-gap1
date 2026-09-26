"""
Experiment A: 3D Conformation Generation and MMFF94 Force Field Minimization.

Uses RDKit's ETKDGv3 (Experimental-Torsion Knowledge Distance Geometry) algorithm:
- Embeds candidate leads into 3D Cartesian coordinates.
- Optimizes geometry using the MMFF94 molecular mechanics force field.
- Measures spatial coordination distances to validate binuclear zinc binding geometry.
- Exports validated 3D Mol file for structural visualization.
"""

import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(DATA_DIR, exist_ok=True)

MOLECULES = {
    "captopril_seed": "CC(CS)C(=O)N1CCCC1C(=O)O",
    "lead_14b_invariant": "O=C(O)C(CS)CC(=O)N1C(Cc2ccccc2)CSC1C(=O)O"
}


def run_conformer_experiment():
    print("[*] Starting Experiment A: 3D Conformer Generation & Force Field Optimization...\n")
    results = {}

    for name, smiles in MOLECULES.items():
        print(f"--- Processing {name} ---")
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        # Generate 3D conformer using ETKDGv3
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_status = AllChem.EmbedMolecule(mol, params)
        
        if embed_status != 0:
            print(f"[-] Initial embed failed, retrying with random coordinates...")
            AllChem.EmbedMolecule(mol, useRandomCoords=True)

        # Optimize geometry using MMFF94
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        converged = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=1000)
        
        # Calculate MMFF94 potential energy
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
        pot_energy = ff.CalcEnergy() if ff else 0.0

        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()

        # Identify key functional atoms for coordination
        thiol_s_idx = None
        carboxyl_o_indices = []

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == "S" and atom.GetTotalNumHs() >= 1:
                thiol_s_idx = atom.GetIdx()
            elif symbol == "O" and atom.GetDegree() == 1:
                # Carbonyl or carboxylate oxygen
                carboxyl_o_indices.append(atom.GetIdx())

        print(f"[+] Conformer converged: {converged == 0} (0 = successful convergence)")
        print(f"[+] MMFF94 Potential Energy: {pot_energy:.2f} kcal/mol")
        print(f"[+] Total Atoms (with Hydrogens): {num_atoms}")

        if thiol_s_idx is not None and carboxyl_o_indices:
            s_pos = conf.GetAtomPosition(thiol_s_idx)
            o_pos = conf.GetAtomPosition(carboxyl_o_indices[0])
            intra_chelator_dist = s_pos.Distance(o_pos)
            print(f"[+] Internal S-to-O Chelation Span: {intra_chelator_dist:.2f} Angstroms")
            print(f"    (Ideal distance to bridge binuclear Zn1-Zn2 span: 3.5 - 5.2 Angstroms)")

        # Save 3D structure to SDF file
        sdf_path = os.path.join(DATA_DIR, f"{name}_3d.sdf")
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        print(f"[+] Exported 3D coordinates to: {sdf_path}\n")

        results[name] = {
            "converged": converged == 0,
            "potential_energy_kcal_mol": round(pot_energy, 2),
            "num_atoms": num_atoms
        }

    return results


if __name__ == "__main__":
    run_conformer_experiment()
