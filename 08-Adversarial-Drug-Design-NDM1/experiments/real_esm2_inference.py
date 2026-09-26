"""
Real ESM-2 Zero-Shot Mutational Fitness Scoring for NDM-1 Active Site.

Runs authentic masked language model forward passes using Meta's ESM-2
(facebook/esm2_t6_8M_UR50D) on the RTX 5070 Ti GPU:
    Delta_Fitness = log P(mutant_aa | context) - log P(wildtype_aa | context)

Evaluates active site hotspots: Met67, Lys211, Asn220, as well as invariant catalytic residues:
His120, His122, Asp124, His189, Cys208, His250.
"""

import json
import os
import sys
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pathogen.ndm1_target import NDM1_WT_SEQUENCE, HOTSPOT_RESIDUES, AMINO_ACIDS

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(DATA_DIR, exist_ok=True)

# Comprehensive active site positions in mature NDM-1 sequence
# (1-indexed based on mature NDM-1 numbering from PDB: 3SPU)
ACTIVE_SITE_TARGETS = {
    # Clinical resistance hotspots on flexible loops:
    67: 'M',   # Met67 (L3 loop roof)
    211: 'K',  # Lys211 (L10 loop entrance)
    220: 'N',  # Asn220 (L10 loop zinc anchor)
    # Catalytic invariant binuclear zinc coordinating residues:
    120: 'H',  # Zn1 coordinating histidine
    122: 'H',  # Zn1 coordinating histidine
    124: 'D',  # Zn2 coordinating aspartate
    189: 'H',  # Zn1 coordinating histidine
    208: 'C',  # Zn2 coordinating cysteine
    250: 'H'   # Zn2 coordinating histidine
}


def run_live_esm2_mutational_scan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing real ESM-2 model ({MODEL_NAME}) on {device.upper()}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    seq = NDM1_WT_SEQUENCE
    print(f"[+] Loaded NDM-1 wild-type sequence (Length: {len(seq)} residues)")

    full_results = {}
    csv_rows = ["position,wt_amino_acid,mut_amino_acid,is_catalytic_core,delta_fitness,is_viable"]

    for pos, wt_aa in ACTIVE_SITE_TARGETS.items():
        is_catalytic = pos in [120, 122, 124, 189, 208, 250]
        site_type = "CATALYTIC CORE" if is_catalytic else "RESISTANCE HOTSPOT"

        # Locate residue in mature sequence
        seq_idx = pos - 1
        if seq_idx >= len(seq) or seq[seq_idx] != wt_aa:
            candidates = [i for i, aa in enumerate(seq) if aa == wt_aa]
            seq_idx = min(candidates, key=lambda i: abs(i - (pos - 1)))

        print(f"\n--- Scanning Position {pos} ({wt_aa}) [{site_type}] (seq_index: {seq_idx}) ---")
        
        # Mask the target residue
        masked_seq = list(seq)
        masked_seq[seq_idx] = tokenizer.mask_token
        masked_seq_str = "".join(masked_seq)

        inputs = tokenizer(masked_seq_str, return_tensors="pt").to(device)
        mask_token_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        with torch.no_grad():
            logits = model(**inputs).logits
            token_logits = logits[0, mask_token_idx, :].squeeze(0)
            log_probs = torch.log_softmax(token_logits, dim=-1)

        wt_token_id = tokenizer.convert_tokens_to_ids(wt_aa)
        wt_log_prob = log_probs[wt_token_id].item()

        pos_results = {}
        for aa in AMINO_ACIDS:
            if aa == wt_aa:
                continue
            mut_token_id = tokenizer.convert_tokens_to_ids(aa)
            mut_log_prob = log_probs[mut_token_id].item()
            delta_fitness = mut_log_prob - wt_log_prob
            is_viable = delta_fitness > -1.5
            pos_results[aa] = {
                "delta_fitness": round(delta_fitness, 3),
                "is_viable": is_viable
            }
            csv_rows.append(f"{pos},{wt_aa},{aa},{is_catalytic},{delta_fitness:.4f},{is_viable}")

        # Sort mutations by fitness
        sorted_muts = sorted(pos_results.items(), key=lambda x: x[1]["delta_fitness"], reverse=True)
        
        viable_count = sum(1 for m, d in sorted_muts if d["is_viable"])
        print(f"[+] Total Viable Mutations (Delta_Fitness > -1.5): {viable_count} / 19")
        
        print("  Top 3 Candidates:")
        for mut_aa, d in sorted_muts[:3]:
            status = "VIABLE" if d["is_viable"] else "PRUNED"
            print(f"    - {wt_aa}{pos}{mut_aa}: Delta_Fitness = {d['delta_fitness']:+.3f} [{status}]")

        print("  Bottom 2 Destabilizing Candidates:")
        for mut_aa, d in sorted_muts[-2:]:
            status = "PRUNED" if not d["is_viable"] else "VIABLE"
            print(f"    - {wt_aa}{pos}{mut_aa}: Delta_Fitness = {d['delta_fitness']:+.3f} [{status}]")

        full_results[f"{wt_aa}{pos}"] = {
            "position": pos,
            "wt_aa": wt_aa,
            "is_catalytic_core": is_catalytic,
            "viable_mutations_count": viable_count,
            "mutations": pos_results
        }

    # Save to disk
    json_path = os.path.join(DATA_DIR, "real_esm2_dms_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    csv_path = os.path.join(DATA_DIR, "real_esm2_dms_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows))

    print(f"\n[+] Full Active-Site ESM-2 Mutational Scan complete!")
    print(f"[+] Saved JSON results to: {json_path}")
    print(f"[+] Saved CSV dataset to: {csv_path}")

    return full_results


if __name__ == "__main__":
    run_live_esm2_mutational_scan()
