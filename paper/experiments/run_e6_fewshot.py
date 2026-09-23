# -*- coding: utf-8 -*-
"""
E6: 小样本（降低训练比例）多种子补跑 —— 对应论文表9
协议（与文中其他多种子补充实验一致）:
  - 固定报告级划分 seed=42: 72 训练报告 / 15 验证 / 16 测试(180节点), 与表5/7/8同一测试集
  - 每个训练比例 r in {0.1,0.2,0.3,0.5,0.7}, 每个随机种子 s in {42,43,44}:
      * 用 RandomState(s) 生成的固定排列, 取前 int(72*r) 份训练报告
        (同一种子内各比例的子集嵌套: 10% ⊂ 20% ⊂ 30% ⊂ 50% ⊂ 70%)
      * torch.manual_seed(s) 控制初始化与批顺序, 完整重训
        (AdamW lr=1e-3 wd=5e-4, batch=16, ≤200轮, 验证损失早停 patience=30, 恢复最优验证状态)
      * 在固定测试集上评估
  - 报告每(模型, 比例)的逐种子结果与 F1 均值±标准差
输出: e6_fewshot_multiseed.json
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiments import load_graphs, split, train_one, evaluate, MODELS, OUT_DIR

RATIOS = [0.1, 0.2, 0.3, 0.5, 0.7]
SEEDS = [42, 43, 44]


def main():
    graphs = load_graphs()
    tr_full, va, te = split(graphs, 42)          # 固定划分, 与表5/7/8一致
    n_tr = len(tr_full)
    print(f'n_train={n_tr} n_val={len(va)} n_test={len(te)} '
          f'test_nodes={sum(g.num_nodes for g in te)}')

    res = {
        'protocol': {
            'split_seed': 42, 'n_train': n_tr, 'n_val': len(va), 'n_test': len(te),
            'test_nodes': sum(g.num_nodes for g in te),
            'test_pos': sum(int(g.y.sum()) for g in te),
            'ratios': RATIOS, 'seeds': SEEDS,
            'subsample': 'RandomState(seed)排列取前int(n*r)份训练报告, 同种子内各比例嵌套',
            'training': 'AdamW lr=1e-3 wd=5e-4, batch=16, max200ep, val-loss早停pat=30, 恢复最优状态',
        },
        'results': {},
    }

    for name in MODELS:
        res['results'][name] = {}
        for r in RATIOS:
            k = int(n_tr * r)
            runs = []
            for s in SEEDS:
                perm = np.random.RandomState(s).permutation(n_tr)
                tr = [tr_full[i] for i in perm[:k]]
                m, dt = train_one(name, 5, tr, va, s)
                ev = evaluate(m, te)
                ev.update(seed=s, time_s=dt, n_train_graphs=k)
                runs.append(ev)
                print(f'E6 {name} r={r} seed{s}: f1={ev["f1"]:.4f} '
                      f'acc={ev["acc"]:.4f} auc={ev["auc"]:.4f}', flush=True)
            f1s = [x['f1'] for x in runs]
            res['results'][name][str(r)] = {
                'runs': runs,
                'n_train_graphs': k,
                'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
                'acc_mean': float(np.mean([x['acc'] for x in runs])),
                'auc_mean': float(np.nanmean([x['auc'] for x in runs])),
            }
        with open(os.path.join(OUT_DIR, 'e6_fewshot_multiseed.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=float)
        print(f'[SAVED] e6_fewshot_multiseed.json ({name} done)', flush=True)

    print('E6 ALL DONE', flush=True)


if __name__ == '__main__':
    main()
