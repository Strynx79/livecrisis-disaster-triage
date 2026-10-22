"""Post-training evals for distilbert + bert_base (whatever has preds).

Writes tagged artifacts: training_curves_<tag>.png, confusion_matrix_<tag>.png,
error_analysis_<tag>.md. Rebuilds comparison_table with all models present.
Never crashes the night: skips missing inputs.
"""
import json
import random
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS, DATA = ROOT / 'results', ROOT / 'data'
TAGS = ['distilbert', 'bert_base']


def main():
    summary = json.loads((ROOT / 'data/splits/split_summary.json').read_text())
    labels = summary['labels']
    test_ids = json.loads((ROOT / 'data/splits/test_ids.json').read_text())
    pooled = pd.read_parquet(DATA / 'humaid_pooled.parquet')
    tmap = {str(i): (t, e) for i, t, e in
            zip(pooled['tweet_id'].astype(str), pooled['text'], pooled['event'])}

    for tag in TAGS:
        hist_p = RESULTS / f'training_history_{tag}.json'
        if hist_p.exists():
            h = json.loads(hist_p.read_text())
            rows = [r for r in h['history'] if r.get('val_macro_f1') is not None]
            if rows:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax2 = ax.twinx()
                ep = [r['epoch'] for r in rows]
                ax.plot(ep, [r['train_loss'] for r in rows], 'o-', label='train loss')
                ax2.plot(ep, [r['val_macro_f1'] for r in rows], 's-',
                         label='val macro-F1')
                ax.set_xlabel('epoch')
                ax.set_ylabel('train loss')
                ax2.set_ylabel('val macro-F1')
                ax.set_title(f'{tag} training (batch {h["batch"]})')
                fig.tight_layout()
                fig.savefig(RESULTS / f'training_curves_{tag}.png', dpi=150)
                print(f'saved training_curves_{tag}.png', flush=True)

        pred_p = RESULTS / f'{tag}_test_preds.pt'
        if not pred_p.exists():
            continue
        d = torch.load(pred_p, map_location='cpu', weights_only=True)
        yt, yp = d['test_true'], d['test_pred']
        cm = confusion_matrix(yt, yp, labels=list(range(len(labels))))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cm)
        short = [l[:18] for l in labels]
        ax.set_xticks(range(len(labels)), short, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(labels)), short, fontsize=7)
        ax.set_xlabel('predicted')
        ax.set_ylabel('true')
        ax.set_title(f'{tag} confusion matrix (test, n={len(yt)}, unseen events)')
        fig.tight_layout()
        fig.savefig(RESULTS / f'confusion_matrix_{tag}.png', dpi=150)
        print(f'saved confusion_matrix_{tag}.png', flush=True)

        wrong = [(test_ids[i], labels[yt[i]], labels[yp[i]]) for i in range(len(yt))
                 if yt[i] != yp[i]]
        random.seed(42)
        sample = random.sample(wrong, min(15, len(wrong)))
        lines = [f'# Error analysis — {tag} test errors (15 sampled)',
                 f'_{len(wrong)} misclassified of {len(yt)} test examples._', '']
        for tid, true, pred in sample:
            text, ev = tmap.get(tid, ('?', '?'))
            guess = ('short/ambiguous text' if len(text) < 80 else
                     'overlapping humanitarian categories' if true.split('_')[0]
                     in pred.split('_') else 'possible label noise / genuinely ambiguous')
            lines += [f'## true={true} pred={pred} (event: {ev})',
                      f'> {text}', f'guess: {guess}', '']
        (RESULTS / f'error_analysis_{tag}.md').write_text('\n'.join(lines))
        print(f'saved error_analysis_{tag}.md ({len(wrong)} total errors)', flush=True)

    # --- comparison table (csv + md), all models present ---
    import csv
    mp = RESULTS / 'metrics.json'
    m = json.loads(mp.read_text()) if mp.exists() else {}
    rows = []
    for r in list(csv.DictReader((RESULTS / 'comparison_table.csv').open())):
        if r['model'] != 'distilbert':  # rebuilt below with single-ex latency
            rows.append(r)
    for tag in TAGS:
        if f'{tag}_test' in m:
            r = m[f'{tag}_test']
            rows.append({'model': tag, 'split': 'test', 'accuracy': r['accuracy'],
                         'macro_f1': r['macro_f1'], 'weighted_f1': r['weighted_f1'],
                         'latency_ms_per_example': r.get(
                             'latency_ms_per_example_single', '')})
    with (RESULTS / 'comparison_table.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = ['| model | split | accuracy | macro-F1 | weighted-F1 | latency ms/ex |',
          '|---|---|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['model']} | {r['split']} | {float(r['accuracy']):.4f} | "
                  f"{float(r['macro_f1']):.4f} | {float(r['weighted_f1']):.4f} | "
                  f"{r.get('latency_ms_per_example', '')} |")
    (RESULTS / 'comparison_table.md').write_text('\n'.join(md) + '\n')
    print('saved comparison_table.md', flush=True)


if __name__ == '__main__':
    main()
