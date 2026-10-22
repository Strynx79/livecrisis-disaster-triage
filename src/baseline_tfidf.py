"""TF-IDF + Logistic Regression baseline (CPU). Saves metrics + confusion matrix."""
import json
import time
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, SPLITS, RESULTS = ROOT / 'data', ROOT / 'data' / 'splits', ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
SEED = 42


def load(name):
    ids = set(json.loads((SPLITS / f'{name}_ids.json').read_text()))
    df = pd.read_parquet(DATA / 'humaid_pooled.parquet')
    return df[df['tweet_id'].astype(str).isin(ids)].reset_index(drop=True)


def main():
    train, val, test = load('train'), load('val'), load('test')
    labels = json.loads((SPLITS / 'split_summary.json').read_text())['labels']
    print(f'train={len(train)} val={len(val)} test={len(test)} classes={len(labels)}', flush=True)

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    Xtr = vec.fit_transform(train['text'])
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                             random_state=SEED)
    t0 = time.time()
    clf.fit(Xtr, train['class_label'])
    print(f'fitted in {time.time() - t0:.1f}s', flush=True)

    out, table = {}, []
    for name, df in [('val', val), ('test', test)]:
        X = vec.transform(df['text'])
        t0 = time.time()
        pred = clf.predict(X)
        lat_ms = (time.time() - t0) / len(df) * 1000
        y = df['class_label']
        p, r, f, _ = precision_recall_fscore_support(
            y, pred, labels=labels, zero_division=0)
        row = {'accuracy': accuracy_score(y, pred),
               'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
               'weighted_f1': f1_score(y, pred, average='weighted', zero_division=0),
               'latency_ms_per_example': lat_ms,
               'per_class': {l: {'precision': float(p[i]), 'recall': float(r[i]),
                                 'f1': float(f[i])} for i, l in enumerate(labels)}}
        out[f'tfidf_lr_{name}'] = row
        table.append({'model': 'tfidf_lr', 'split': name, **{k: v for k, v in row.items()
                                                             if k != 'per_class'}})
        print(f'{name}: acc={row["accuracy"]:.4f} macroF1={row["macro_f1"]:.4f} '
              f'lat={lat_ms:.2f}ms/ex', flush=True)

    # Confusion matrix on TEST.
    cm = confusion_matrix(test['class_label'], clf.predict(vec.transform(test['text'])),
                          labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title('TF-IDF+LR confusion matrix (test, n=12872, unseen events)')
    fig.tight_layout()
    fig.savefig(RESULTS / 'confusion_matrix_tfidf_lr.png', dpi=150)

    prev = {}
    mp = RESULTS / 'metrics.json'
    if mp.exists():
        prev = json.loads(mp.read_text())
    prev.update(out)
    mp.write_text(json.dumps(prev, indent=1))
    pd.DataFrame(table).to_csv(RESULTS / 'comparison_table.csv', index=False)
    print('saved metrics.json, comparison_table.csv, confusion_matrix_tfidf_lr.png', flush=True)


if __name__ == '__main__':
    main()
