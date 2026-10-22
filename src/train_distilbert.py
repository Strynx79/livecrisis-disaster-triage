"""Overnight: fine-tune distilbert-base-uncased on HumAID event split.

Autonomous-OOM design: tries (batch 16 x accum 2), falls back to (8 x 4)
on CUDA OOM. Checkpoints every epoch, best by VAL macro-F1 (9 classes
present in val). Saves best to models/distilbert/ + history + test metrics.
"""
import json
import time
from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA, SPLITS, RESULTS, MODELS = (ROOT / 'data', ROOT / 'data' / 'splits',
                                 ROOT / 'results', ROOT / 'models')
for d in (RESULTS, MODELS, ROOT / 'logs'):
    d.mkdir(exist_ok=True)

SEED, MAX_LEN, LR, EPOCHS = 42, 128, 2.0e-5, 4
MODEL_NAME = 'distilbert-base-uncased'
TAG = 'distilbert'  # overridden by train_bert.py
ATTEMPTS = [(16, 2), (8, 4)]  # (per_device_batch, grad_accum) effective 32


class WeightedTrainer(Trainer):
    """Trainer with class-weighted CE loss.

    NOTE: must return (loss, outputs) when return_outputs=True — Trainer's
    eval loop unpacks strictly; a bare tensor crashes eval with
    "TypeError: iteration over a 0-d tensor". Never pop labels:
    prediction_step reads inputs['labels'] after this call.
    """

    def __init__(self, *a, class_weights=None, **k):
        super().__init__(*a, **k)
        self.cw = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_ = inputs['labels']
        out = model(input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'))
        loss = torch.nn.functional.cross_entropy(
            out.logits, labels_, weight=self.cw.to(out.logits.device))
        return (loss, out) if return_outputs else loss


def load(name):
    ids = set(json.loads((SPLITS / f'{name}_ids.json').read_text()))
    df = pd.read_parquet(DATA / 'humaid_pooled.parquet')
    df = df[df['tweet_id'].astype(str).isin(ids)].reset_index(drop=True)
    return df


def main():
    summary = json.loads((SPLITS / 'split_summary.json').read_text())
    labels = summary['labels']
    lid = {l: i for i, l in enumerate(labels)}
    train_df, val_df, test_df = load('train'), load('val'), load('test')
    val_labels_present = sorted(set(val_df['class_label'].unique().tolist()))
    print(f'train={len(train_df)} val={len(val_df)} test={len(test_df)} '
          f'val_classes={len(val_labels_present)}/10', flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok_fn(b):
        return tok(b['text'], truncation=True, max_length=MAX_LEN)

    def to_ds(df):
        d = Dataset.from_pandas(df[['text', 'class_label']].rename(
            columns={'class_label': 'label_str'}))
        d = d.map(lambda r: {'label': lid[r['label_str']]}, remove_columns=['label_str'])
        return d.map(tok_fn, batched=True, remove_columns=['text'])

    train_ds, val_ds = to_ds(train_df), to_ds(val_df)

    cw = torch.tensor(
        [len(train_df) / (len(labels) * (train_df['class_label'] == l).sum())
         for l in labels], dtype=torch.float)

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        lab = p.label_ids
        return {'accuracy': accuracy_score(lab, preds),
                'val_macro_f1': f1_score(lab, preds, average='macro', zero_division=0),
                'val_weighted_f1': f1_score(lab, preds, average='weighted',
                                           zero_division=0)}

    trainer, used = None, None
    for pb, ga in ATTEMPTS:
        try:
            print(f'ATTEMPT per_device={pb} accum={ga} (eff {pb * ga})', flush=True)
            torch.cuda.empty_cache()
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=len(labels))
            args = TrainingArguments(
                output_dir=str(MODELS / f'{TAG}_checkpoints'),
                seed=SEED, per_device_train_batch_size=pb,
                per_device_eval_batch_size=32, gradient_accumulation_steps=ga,
                num_train_epochs=EPOCHS, learning_rate=LR, weight_decay=0.01,
                fp16=True, eval_strategy='epoch', save_strategy='epoch',
                load_best_model_at_end=True, metric_for_best_model='val_macro_f1',
                greater_is_better=True, save_total_limit=2, report_to='none',
                log_level='warning')
            trainer = WeightedTrainer(
                model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                compute_metrics=compute_metrics, class_weights=cw)
            trainer.train()
            used = (pb, ga)
            history = [{'epoch': h['epoch'], 'train_loss': h.get('loss'),
                        'val_macro_f1': h.get('eval_val_macro_f1'),
                        'val_accuracy': h.get('eval_accuracy')}
                       for h in trainer.state.log_history if 'epoch' in h]
            break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'OOM at batch {pb}: halving. {e}', flush=True)
                del trainer
                torch.cuda.empty_cache()
                continue
            raise
    if trainer is None:
        (ROOT / 'STATUS.md').write_text(
            (ROOT / 'STATUS.md').read_text() + '\nDistilBERT FAILED twice (OOM). Skipped.\n')
        raise SystemExit('DistilBERT failed on all batch configs')

    best = trainer.state.best_metric
    print(f'done. batch={used} best_val_macroF1={best:.4f}', flush=True)
    trainer.save_model(str(MODELS / TAG))
    tok.save_pretrained(str(MODELS / TAG))
    (MODELS / TAG / 'label_map.json').write_text(json.dumps(lid, indent=1))
    (RESULTS / f'training_history_{TAG}.json').write_text(json.dumps(
        {'batch': used, 'history': history}, indent=1))

    # Test eval (best model, fp32, batched) + honest single-example latency.
    test_ds = to_ds(test_df)
    t0 = time.time()
    pred = trainer.predict(test_ds)
    test_lat_batch = (time.time() - t0) / len(test_df) * 1000
    yp = pred.predictions.argmax(-1)
    yt = test_df['class_label'].map(lid).tolist()
    row = {'accuracy': accuracy_score(yt, yp),
           'macro_f1': f1_score(yt, yp, average='macro', zero_division=0),
           'weighted_f1': f1_score(yt, yp, average='weighted', zero_division=0),
           'latency_ms_per_example_batch': test_lat_batch}
    # single-example latency on 100 samples
    model = trainer.model.eval()
    samp = test_df['text'].tolist()[:100]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    with torch.no_grad():
        for t in samp:
            inp = tok(t, return_tensors='pt', truncation=True,
                      max_length=MAX_LEN).to(model.device)
            model(**inp)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    row['latency_ms_per_example_single'] = (time.time() - t0) / 100 * 1000
    print(f'test: acc={row["accuracy"]:.4f} macroF1={row["macro_f1"]:.4f} '
          f'lat_single={row["latency_ms_per_example_single"]:.1f}ms', flush=True)

    mp = RESULTS / 'metrics.json'
    prev = json.loads(mp.read_text()) if mp.exists() else {}
    prev[f'{TAG}_test'] = row
    prev[f'{TAG}_val_best'] = {'val_macro_f1': best, 'batch': list(used)}
    mp.write_text(json.dumps(prev, indent=1))

    torch.save({'test_true': yt, 'test_pred': yp.tolist()},
               RESULTS / f'{TAG}_test_preds.pt')
    print('saved model + metrics + preds', flush=True)


if __name__ == '__main__':
    main()
