"""Smoke test for the compute_loss fix: runs trainer.evaluate() on 64 examples,
exercising prediction_step -> compute_loss(return_outputs=True), the exact
crash site from the overnight run."""
import sys
import json
from pathlib import Path
import torch
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, DataCollatorWithPadding)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from train_distilbert import WeightedTrainer, SEED, MAX_LEN, MODEL_NAME  # noqa

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary = json.loads((ROOT / 'data/splits/split_summary.json').read_text())
    labels = summary['labels']
    lid = {l: i for i, l in enumerate(labels)}
    df = pd.read_parquet(ROOT / 'data/humaid_pooled.parquet').head(64)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = Dataset.from_pandas(pd.DataFrame(
        {'text': df['text'], 'label': df['class_label'].map(lid)}))
    ds = ds.map(lambda b: tok(b['text'], truncation=True, max_length=MAX_LEN),
                batched=True, remove_columns=['text'])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels))
    args = TrainingArguments(output_dir=str(ROOT / 'logs/smoke_out'), seed=SEED,
                             per_device_eval_batch_size=16, report_to='none',
                             fp16=True)
    cw = torch.ones(len(labels))
    trainer = WeightedTrainer(model=model, args=args, eval_dataset=ds,
                              processing_class=tok,
                              data_collator=DataCollatorWithPadding(tok),
                              class_weights=cw)
    m = trainer.evaluate()
    print('SMOKE_EVAL_OK', {k: round(float(v), 4) for k, v in m.items()
                            if 'loss' in k or 'runtime' in k}, flush=True)


if __name__ == '__main__':
    main()
