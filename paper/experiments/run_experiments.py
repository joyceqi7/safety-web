# -*- coding: utf-8 -*-
"""
补实验总控脚本：复现 + 多种子 + 特征消融 + 模型消融 + 鲁棒性 + 规则基线
运行环境: paper_env (torch cpu + torch_geometric + sklearn)
所有结果写入 exp_results/*.json
"""
import sys, os, json, time, copy
import numpy as np
import torch
import torch.nn.functional as F_fn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
sys.path.insert(0, os.path.join(_ROOT, 'model_source'))
from gatv2_base import GATv2Base
from hierarchical_gatv2 import HierarchicalGATv2
from uncertainty_gatv2 import UncertaintyGATv2, UncertaintyLoss
from multi_scale_gatv2 import MultiScaleGATv2
from dynamic_temporal_gatv2 import DynamicTemporalGATv2
from gatv2_transformer import GATv2Transformer

BASE = os.path.join(_ROOT, 'processed_data')
OUT_DIR = os.path.join(_HERE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cpu')

# ---------------- 数据 ----------------
def normalize(f):
    return [(v - 1.0) / 4.0 for v in f]

def load_graphs():
    with open(os.path.join(BASE, 'extracted_data', '_all_extractions.json'),
              encoding='utf-8') as f:
        all_ext = json.load(f)
    graphs = []
    for ext in all_ext:
        nodes, edges = ext.get('nodes', []), ext.get('edges', [])
        if len(nodes) < 2:
            continue
        id2idx = {n['id']: i for i, n in enumerate(nodes)}
        x = torch.tensor([normalize(n['features']) for n in nodes], dtype=torch.float)
        y = torch.tensor([1 if n['is_high_risk'] else 0 for n in nodes], dtype=torch.long)
        el, ew = [], []
        for e in edges:
            s, t = id2idx.get(e['source']), id2idx.get(e['target'])
            if s is not None and t is not None:
                el.append([s, t]); ew.append(e.get('confidence', 0.5))
        if el:
            ei = torch.tensor(el, dtype=torch.long).t().contiguous()
            ew = torch.tensor(ew, dtype=torch.float)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long); ew = torch.zeros(0)
        g = Data(x=x, y=y, edge_index=ei, edge_weight=ew, num_nodes=len(nodes))
        g.raw_features = torch.tensor([n['features'] for n in nodes], dtype=torch.float)
        graphs.append(g)
    return graphs

def make_3feat(graphs):
    """去掉 frequency(idx0) 与 severity(idx1)，保留 D/C/V (idx 2,3,4)"""
    out = []
    for g in graphs:
        g2 = copy.copy(g)
        g2.x = g.x[:, [2, 3, 4]].contiguous()
        out.append(g2)
    return out

def strip_edges(graphs):
    out = []
    for g in graphs:
        g2 = copy.copy(g)
        g2.edge_index = torch.zeros((2, 0), dtype=torch.long)
        g2.edge_weight = torch.zeros(0)
        out.append(g2)
    return out

def split(graphs, seed=42, tr=0.7, va=0.15):
    np.random.seed(seed)
    n = len(graphs)
    idx = np.random.permutation(n)
    tr_end = int(n * tr); va_end = int(n * (tr + va))
    return ([graphs[i] for i in idx[:tr_end]],
            [graphs[i] for i in idx[tr_end:va_end]],
            [graphs[i] for i in idx[va_end:]])

# ---------------- 模型 ----------------
class HierNoContext(HierarchicalGATv2):
    """消融: 移除图级上下文, 仅用顶层节点表示分类"""
    def __init__(self, in_dim=5, hidden_dim=128, num_layers=3, num_heads=4,
                 dropout=0.3, num_classes=2, pool_ratio=0.7):
        super().__init__(in_dim, hidden_dim, num_layers, num_heads, dropout,
                         num_classes, pool_ratio)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.bottom_conv(x, edge_index); x1 = self.bottom_norm(x1)
        x1 = F_fn.elu(x1)
        x1 = F_fn.dropout(x1, p=self.dropout, training=self.training)
        x1p = self.bottom_proj(x1)
        x2 = self.mid_conv(x1p, edge_index); x2 = self.mid_norm(x2)
        x2 = F_fn.elu(x2)
        x2 = F_fn.dropout(x2, p=self.dropout, training=self.training)
        x2p = self.mid_proj(x2)
        x3, attn = self.top_conv(x2p, edge_index, return_attention_weights=True)
        self.attention_weights = attn
        x3 = self.top_norm(x3)
        return self.cls(x3)

MODELS = {
    'GATv2': GATv2Base,
    'HierarchicalGATv2': HierarchicalGATv2,
    'UncertaintyGATv2': UncertaintyGATv2,
    'DynamicTemporalGATv2': DynamicTemporalGATv2,
    'GATv2Transformer': GATv2Transformer,
    'MultiScaleGATv2': MultiScaleGATv2,
}

# ---------------- 训练与评估 ----------------
HP = dict(hidden_dim=128, num_layers=3, num_heads=4, dropout=0.3)
LR, WD, BS, MAX_EP, PAT = 1e-3, 5e-4, 16, 200, 30

def forward_model(model, batch, training):
    out = model(batch)
    if isinstance(model, UncertaintyGATv2):
        return out  # (logits, unc)
    return out

def train_one(name, in_dim, train_gs, val_gs, seed, max_ep=MAX_EP):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MODELS[name](in_dim=in_dim, **HP).to(DEVICE)
    is_unc = isinstance(model, UncertaintyGATv2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ce = torch.nn.CrossEntropyLoss()
    uloss = UncertaintyLoss()
    tl = DataLoader(train_gs, batch_size=BS, shuffle=True)
    vl = DataLoader(val_gs, batch_size=BS, shuffle=False)
    best_val, best_state, bad = 1e9, None, 0
    t0 = time.time()
    for ep in range(max_ep):
        model.train()
        for b in tl:
            opt.zero_grad()
            out = forward_model(model, b, True)
            if is_unc:
                loss = uloss(out, b.y)
            else:
                loss = ce(out, b.y)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl_loss, n_node = 0.0, 0
            for b in vl:
                out = forward_model(model, b, False)
                if is_unc:
                    l = ce(out[0], b.y)
                else:
                    l = ce(out, b.y)
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
    dt = time.time() - t0
    return model, dt

@torch.no_grad()
def evaluate(model, graphs, feat_noise=None, edge_keep=None, rs=None):
    model.eval()
    if rs is not None:
        torch.manual_seed(rs); np.random.seed(rs)
    gs = []
    for g in graphs:
        g2 = copy.copy(g)
        x = g2.x
        if feat_noise is not None and feat_noise > 0:
            x = x + torch.randn_like(x) * feat_noise
        ei = g2.edge_index
        if edge_keep is not None and edge_keep < 1.0 and ei.size(1) > 0:
            m = int(round(ei.size(1) * edge_keep))
            perm = torch.randperm(ei.size(1))[:m]
            ei = ei[:, perm]
        g2.x, g2.edge_index = x, ei
        gs.append(g2)
    b = Batch.from_data_list(gs)
    out = model(b)
    if isinstance(model, UncertaintyGATv2):
        logits = out[0]
    else:
        logits = out
    prob = torch.softmax(logits, dim=-1)[:, 1]
    pred = logits.argmax(dim=-1)
    y = b.y
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    return {
        'acc': (pred == y).float().mean().item(),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'auc': roc_auc_score(y, prob.numpy()) if len(set(y.tolist())) > 1 else float('nan'),
        'n_nodes': int(y.numel()),
        'n_pos': int(y.sum().item()),
    }

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def save(name, obj):
    with open(os.path.join(OUT_DIR, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    print(f'[SAVED] {name}')

# ---------------- 实验主流程 ----------------
def main():
    graphs = load_graphs()
    print(f'graphs={len(graphs)} nodes={sum(g.num_nodes for g in graphs)} '
          f'edges={sum(g.edge_index.size(1) for g in graphs)}')

    # ---- E0: 规则基线 + 数据统计 ----
    tr, va, te = split(graphs, 42)
    te_nodes = sum(g.num_nodes for g in te)
    te_pos = sum(int(g.y.sum()) for g in te)
    all_pos = sum(int(g.y.sum()) for g in graphs)
    rule = {
        'total_graphs': len(graphs),
        'total_nodes': sum(g.num_nodes for g in graphs),
        'total_edges': sum(g.edge_index.size(1) for g in graphs),
        'avg_nodes': sum(g.num_nodes for g in graphs) / len(graphs),
        'avg_edges': sum(g.edge_index.size(1) for g in graphs) / len(graphs),
        'n_train_graphs': len(tr), 'n_val_graphs': len(va), 'n_test_graphs': len(te),
        'test_nodes': te_nodes, 'test_pos': te_pos,
        'high_risk_ratio': all_pos / sum(g.num_nodes for g in graphs),
    }
    # 规则基线: severity>=4 或 (severity>=3 且 frequency>=4), 作用在原始1-5量表
    ys, ps = [], []
    for g in graphs:
        rf = g.raw_features
        rule_pred = ((rf[:, 1] >= 4) | ((rf[:, 1] >= 3) & (rf[:, 0] >= 4))).long()
        ys.append(g.y); ps.append(rule_pred)
    ys, ps = torch.cat(ys), torch.cat(ps)
    from sklearn.metrics import precision_score, recall_score, f1_score
    rule['rule_baseline'] = {
        'acc': (ps == ys).float().mean().item(),
        'precision': precision_score(ys, ps, zero_division=0),
        'recall': recall_score(ys, ps, zero_division=0),
        'f1': f1_score(ys, ps, zero_division=0),
    }
    save('e0_stats_rule.json', rule)
    print('E0 done:', json.dumps(rule['rule_baseline']))

    # ---- E1: 5特征 固定划分 多种子 (6模型 x seeds 42/43/44) ----
    e1 = {}
    for name in MODELS:
        rows = []
        for seed in (42, 43, 44):
            tr, va, te = split(graphs, seed)
            m, dt = train_one(name, 5, tr, va, seed)
            r = evaluate(m, te)
            r['seed'] = seed; r['time_s'] = dt; r['params'] = count_params(m)
            rows.append(r)
            print(f"E1 {name} seed{seed}: f1={r['f1']:.4f} acc={r['acc']:.4f} auc={r['auc']:.4f}")
        e1[name] = rows
        save('e1_fixed_5feat.json', e1)
        # 保存E1种子42的模型用于鲁棒性评估
        torch.save(m.state_dict(), os.path.join(OUT_DIR, f'm5_{name}.pt'))

    # ---- E2: 3特征(去F/S) 固定划分 多种子 ----
    graphs3 = make_3feat(graphs)
    e2 = {}
    for name in MODELS:
        rows = []
        for seed in (42, 43, 44):
            tr, va, te = split(graphs3, seed)
            m, dt = train_one(name, 3, tr, va, seed)
            r = evaluate(m, te)
            r['seed'] = seed; r['time_s'] = dt; r['params'] = count_params(m)
            rows.append(r)
            print(f"E2 {name} seed{seed}: f1={r['f1']:.4f} acc={r['acc']:.4f} auc={r['auc']:.4f}")
        e2[name] = rows
        save('e2_fixed_3feat.json', e2)
        torch.save(m.state_dict(), os.path.join(OUT_DIR, f'm3_{name}.pt'))

    # ---- E3: 消融 (HierarchicalGATv2) ----
    e3 = {}
    variants = {
        'HierNoContext': (HierNoContext, 5, graphs),
        'HierNoEdges': (HierarchicalGATv2, 5, strip_edges(graphs)),
        'GATv2NoEdges': (GATv2Base, 5, strip_edges(graphs)),
    }
    for vname, (cls, dim, gs) in variants.items():
        rows = []
        for seed in (42, 43, 44):
            tr, va, te = split(gs, seed)
            model_cls_local = cls
            if vname == 'HierNoContext':
                torch.manual_seed(seed); np.random.seed(seed)
                m = HierNoContext(in_dim=dim, **HP)
            else:
                m, dt = None, None
                # 用统一训练函数但替换模型类
                m = train_one_custom(cls, dim, tr, va, seed)
            r = evaluate(m, te); r['seed'] = seed
            rows.append(r)
            print(f"E3 {vname} seed{seed}: f1={r['f1']:.4f} acc={r['acc']:.4f}")
        e3[vname] = rows
        save('e3_ablation.json', e3)

    # ---- E4: 鲁棒性(多种子噪声/删边, 用E1 seed42模型) ----
    e4 = {'noise_5feat': {}, 'edge_5feat': {}, 'noise_3feat': {}}
    tr, va, te = split(graphs, 42)
    tr3, va3, te3 = split(graphs3, 42)
    for name in MODELS:
        m = MODELS[name](in_dim=5, **HP)
        m.load_state_dict(torch.load(os.path.join(OUT_DIR, f'm5_{name}.pt'),
                                      weights_only=True))
        # 噪声
        lv = {}
        for sig in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            fs = [evaluate(m, te, feat_noise=sig, rs=ns)['f1'] for ns in (7, 17, 27)]
            lv[str(sig)] = {'mean': float(np.mean(fs)), 'std': float(np.std(fs))}
        f0, fm = lv['0.0']['mean'], lv['0.3']['mean']
        lv['decay'] = (f0 - fm) / f0 if f0 > 0 else None
        e4['noise_5feat'][name] = lv
        # 删边
        ev = {}
        for keep in (1.0, 0.7, 0.5, 0.3, 0.1):
            fs = [evaluate(m, te, edge_keep=keep, rs=ns)['f1'] for ns in (7, 17, 27)]
            ev[str(keep)] = {'mean': float(np.mean(fs)), 'std': float(np.std(fs))}
        e4['edge_5feat'][name] = ev
        save('e4_robustness.json', e4)
        print(f"E4 5feat {name} noise decay={lv['decay']}")
    for name in MODELS:
        m = MODELS[name](in_dim=3, **HP)
        m.load_state_dict(torch.load(os.path.join(OUT_DIR, f'm3_{name}.pt'),
                                      weights_only=True))
        lv = {}
        for sig in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            fs = [evaluate(m, te3, feat_noise=sig, rs=ns)['f1'] for ns in (7, 17, 27)]
            lv[str(sig)] = {'mean': float(np.mean(fs)), 'std': float(np.std(fs))}
        f0, fm = lv['0.0']['mean'], lv['0.3']['mean']
        lv['decay'] = (f0 - fm) / f0 if f0 > 0 else None
        e4['noise_3feat'][name] = lv
        save('e4_robustness.json', e4)
        print(f"E4 3feat {name} noise decay={lv['decay']}")

    # ---- E5: 五折交叉验证 (报告级分组) 5特征与3特征 ----
    e5 = {'5feat': {}, '3feat': {}}
    for tag, gs in (('5feat', graphs), ('3feat', graphs3)):
        dim = 5 if tag == '5feat' else 3
        for name in MODELS:
            folds = []
            n = len(gs)
            rng = np.random.RandomState(2026)
            perm = rng.permutation(n)
            for k in range(5):
                te_i = perm[k::5]
                rest = np.setdiff1d(perm, te_i)
                va_i = rest[:len(rest)//4]
                tr_i = rest[len(rest)//4:]
                tr = [gs[i] for i in tr_i]; va = [gs[i] for i in va_i]
                te = [gs[i] for i in te_i]
                seed = 200 + k
                m, _ = train_one(name, dim, tr, va, seed)
                r = evaluate(m, te)
                folds.append(r)
                print(f"E5 {tag} {name} fold{k}: f1={r['f1']:.4f}")
            f1s = [r['f1'] for r in folds]; aucs = [r['auc'] for r in folds]
            e5[tag][name] = {
                'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
                'auc_mean': float(np.nanmean(aucs)), 'auc_std': float(np.nanstd(aucs)),
                'folds': folds,
            }
            save('e5_kfold.json', e5)
    print('ALL DONE')

def train_one_custom(cls, in_dim, train_gs, val_gs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = cls(in_dim=in_dim, **HP).to(DEVICE)
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

if __name__ == '__main__':
    main()
