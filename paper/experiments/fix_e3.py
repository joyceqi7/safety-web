# -*- coding: utf-8 -*-
"""修复 E3 的 HierNoContext：上次忘了训练。重新训练并更新 e3_ablation.json"""
import sys, os, json, copy, time
import numpy as np
import torch
from torch_geometric.loader import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'model_source'))
sys.path.insert(0, _HERE)
from run_experiments import (HierNoContext, load_graphs, split, HP, MODELS,
                             evaluate, LR, WD, BS, MAX_EP, PAT, OUT_DIR)

def train_custom(model, train_gs, val_gs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ce = torch.nn.CrossEntropyLoss()
    tl = DataLoader(train_gs, batch_size=BS, shuffle=True)
    vl = DataLoader(val_gs, batch_size=BS, shuffle=False)
    best_val, best_state, bad = 1e9, None, 0
    for ep in range(MAX_EP):
        model.train()
        for b in tl:
            opt.zero_grad()
            loss = ce(model(b), b.y)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl_loss, n_node = 0.0, 0
            for b in vl:
                l = ce(model(b), b.y)
                vl_loss += l.item() * b.num_nodes; n_node += b.num_nodes
            vl_loss /= max(n_node, 1)
        if vl_loss < best_val - 1e-5:
            best_val, bad = vl_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= PAT:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def main():
    graphs = load_graphs()
    rows = []
    for seed in (42, 43, 44):
        tr, va, te = split(graphs, seed)
        m = HierNoContext(in_dim=5, **HP)
        m = train_custom(m, tr, va, seed)
        r = evaluate(m, te); r['seed'] = seed
        rows.append(r)
        print(f"HierNoContext seed{seed}: f1={r['f1']:.4f} acc={r['acc']:.4f}")
    p = os.path.join(OUT_DIR, 'e3_ablation.json')
    with open(p, encoding='utf-8') as f:
        e3 = json.load(f)
    e3['HierNoContext'] = rows
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(e3, f, ensure_ascii=False, indent=1, default=float)
    print('updated e3_ablation.json')

if __name__ == '__main__':
    main()
