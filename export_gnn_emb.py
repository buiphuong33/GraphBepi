import os
import argparse
import torch
import numpy as np
from model import GraphBepi
from dataset import PDB
from torch.utils.data import DataLoader


def load_model(ckpt, device='cpu'):
    m = GraphBepi()
    state = torch.load(ckpt, map_location=device)
    if 'state_dict' in state:
        m.load_state_dict(state['state_dict'])
    else:
        m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m


def export_embeddings(ckpt, root, out_dir='./tabular', split='train', batch=8, gpu=-1):
    device = 'cpu' if gpu == -1 or not torch.cuda.is_available() else f'cuda:{gpu}'
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Loading model {ckpt} on {device}")
    model = load_model(ckpt, device=device)

    dataset = PDB(mode=split, fold=-1, root=root)

    names_out = []
    idxs_out = []
    resn_out = []
    emb_out = []

    # batch by indices from dataset.data to keep names/order
    for start in range(0, len(dataset), batch):
        end = min(start + batch, len(dataset))
        batch_samples = dataset.data[start:end]
        V_list = []
        E_list = []
        for s in batch_samples:
            s.load_feat(root)
            s.load_dssp(root)
            s.load_adj(root)
            V_list.append(torch.cat([s.feat, s.dssp], 1))
            E_list.append(s.edge)
        # move tensors to device
        V_list = [v.to(device) for v in V_list]
        E_list = [e.to(device) for e in E_list]
        with torch.no_grad():
            h_list = model.embed(V_list, E_list)
        for s, h in zip(batch_samples, h_list):
            h_cpu = h.cpu().numpy()
            L = h_cpu.shape[0]
            for i in range(L):
                names_out.append(s.name)
                idxs_out.append(i)
                resn_out.append(s.sequence[i])
                emb_out.append(h_cpu[i])
    emb_arr = np.vstack(emb_out).astype(np.float32)
    names_arr = np.array(names_out, dtype=object)
    idxs_arr = np.array(idxs_out, dtype=np.int32)
    resn_arr = np.array(resn_out, dtype=object)

    out_path = os.path.join(out_dir, f'gnn_{split}.npz')
    np.savez_compressed(out_path, emb=emb_arr, names=names_arr, idxs=idxs_arr, resn=resn_arr)
    print(f"[DONE] Exported GNN embeddings to {out_path} (shape {emb_arr.shape})")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--root', type=str, default='./data/BCE_633')
    parser.add_argument('--out', type=str, default='./tabular')
    parser.add_argument('--split', type=str, default='train', choices=['train','val','test'])
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    export_embeddings(args.ckpt, args.root, args.out, args.split, args.batch, args.gpu)