import os
import re
import pickle as pk
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import esm

from utils import chain, extract_chain, process_chain


def parse_epi_list(epi_str):
    """
    "148_GLN_A, 122_PHE_A" → dict { 'A': set([148, 122]) }
    """
    out = defaultdict(set)
    items = [x.strip() for x in str(epi_str).split(",") if x.strip()]
    for it in items:
        # <resi>_<resn>_<chain>
        parts = it.split("_")
        if len(parts) != 3:
            continue
        resi_s, _, ch = parts
        m = re.match(r"(\d+)", resi_s)
        if not m:
            continue
        out[ch].add(int(m.group(1)))
    return out


def choose_device(gpu):
    if gpu == -1 or not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{gpu}"


def ensure_folders(root):
    for d in ["PDB", "purePDB", "feat", "dssp", "graph"]:
        os.makedirs(os.path.join(root, d), exist_ok=True)


def main(args):
    # cache ESM
    if args.cache:
        os.environ["TORCH_HOME"] = args.cache
        os.makedirs(args.cache, exist_ok=True)

    root = args.root
    ensure_folders(root)

    device = choose_device(args.gpu)
    print(f"[INFO] Device: {device}")

    # load ESM
    if args.esm_size == "3B":
        esm_model, _ = esm.pretrained.esm2_t36_3B_UR50D()
    elif args.esm_size == "150M":
        esm_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    else:
        esm_model, _ = esm.pretrained.esm2_t33_650M_UR50D()

    esm_model = esm_model.to(device).eval()

    # read CSV
    df = pd.read_csv(args.csv)

    # collect (pdb, chain) → epitope residues
    chain_map = defaultdict(set)
    for _, row in df.iterrows():
        pdb = str(row["PDB ID"]).lower().strip()
        epi_map = parse_epi_list(row["Epitope List (residueid_residuename_chain)"])
        for ch, resi_set in epi_map.items():
            chain_map[(pdb, ch)].update(resi_set)

    print(f"[INFO] Total test chains: {len(chain_map)}")

    test_samples = []

    for (pdb, ch), epi_resi in tqdm(chain_map.items(), desc="Building test set"):
        try:
            extract_chain(root, pdb, ch)
        except Exception as e:
            print(f"[WARN] extract_chain failed for {pdb}_{ch}: {e}")
            continue

        obj = chain()
        obj.name = f"{pdb}_{ch}"

        try:
            process_chain(obj, root, obj.name, esm_model, device)
        except Exception as e:
            print(f"[WARN] process_chain failed for {obj.name}: {e}")
            continue

        # label theo idx+1 (giống BCE_633)
        L = len(obj.sequence)
        y = np.zeros(L, dtype=np.float32)
        for i in range(L):
            if (i + 1) in epi_resi:
                y[i] = 1.0

        obj.label = torch.tensor(y)

        if obj.label.sum() == 0:
            print(f"[WARN] {obj.name}: no positive residues after mapping")

        test_samples.append(obj)

    with open(os.path.join(root, "test.pkl"), "wb") as f:
        pk.dump(test_samples, f)

    print(f"[DONE] test.pkl written with {len(test_samples)} chains")
    print(f"[INFO] Dataset root: {root}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Epitope3D CSV path")
    ap.add_argument("--root", required=True, help="Output dataset root")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--esm_size", default="650M", choices=["150M", "650M", "3B"])
    ap.add_argument("--cache", default="/kaggle/working/graphbepi_cache")
    args = ap.parse_args()
    main(args)
