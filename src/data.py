"""Step 1: load HumAID (all events, all splits), pool, split BY EVENT, save.

Outputs:
  data/humaid_pooled.parquet   (tweet_id, text, label, event, src_split)
  data/splits/{train,val,test}_ids.json
  data/splits/split_summary.json (sizes + per-split class distribution)
"""
import json
from pathlib import Path
import pandas as pd
from datasets import load_dataset

SEED = 42
EVENTS = ['hurricane_florence_2018', 'kaikoura_earthquake_2016', 'kerala_floods_2018',
          'hurricane_harvey_2017', 'hurricane_maria_2017', 'midwestern_us_floods_2019',
          'puebla_mexico_earthquake_2017', 'maryland_floods_2018', 'hurricane_irma_2017',
          'ecuador_earthquake_2016', 'cyclone_idai_2019', 'canada_wildfires_2016',
          'italy_earthquake_aug_2016', 'greece_wildfires_2018', 'hurricane_dorian_2019',
          'california_wildfires_2018', 'pakistan_earthquake_2019', 'hurricane_matthew_2016',
          'srilanka_floods_2017']

# Held-out whole events: type-diverse, ~15% each, rare classes covered.
TEST_EVENTS = ['california_wildfires_2018', 'cyclone_idai_2019', 'ecuador_earthquake_2016']
VAL_EVENTS = ['hurricane_florence_2018', 'kaikoura_earthquake_2016', 'canada_wildfires_2016']
# NOTE: maryland_floods_2018 stays in TRAIN — it holds most missing_or_found_people
# examples (rarest class, 358 total). Val therefore has 9/10 classes; checkpoint
# selection uses macro-F1 over classes present in val, missing_or_found is test-only.

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
SPLITS = DATA / 'splits'
SPLITS.mkdir(parents=True, exist_ok=True)


def main():
    frames = []
    for ev in EVENTS:
        for split in ('train', 'dev', 'test'):
            try:
                ds = load_dataset('QCRI/HumAID-events', ev, split=split)
            except Exception as e:
                print(f'skip {ev}/{split}: {e}', flush=True)
                continue
            df = ds.to_pandas()[['tweet_id', 'tweet_text', 'class_label']].copy()
            df['event'] = ev
            df['src_split'] = split
            frames.append(df)
            print(f'{ev}/{split}: {len(df)}', flush=True)
    pooled = pd.concat(frames, ignore_index=True)

    # Clean: strip, drop empties, drop duplicate tweet_ids (keep first).
    pooled['text'] = pooled['tweet_text'].astype(str).str.strip()
    before = len(pooled)
    pooled = pooled[pooled['text'] != '']
    pooled = pooled.drop_duplicates(subset=['tweet_id']).reset_index(drop=True)
    print(f'pooled={len(pooled)} (dropped {before - len(pooled)} empty/dup)', flush=True)
    pooled[['tweet_id', 'text', 'class_label', 'event', 'src_split']].to_parquet(
        DATA / 'humaid_pooled.parquet', index=False)

    labels = sorted(pooled['class_label'].unique().tolist())
    print(f'label set ({len(labels)}): {labels}', flush=True)

    def subset(events):
        return pooled[pooled['event'].isin(events)]

    splits = {'train': subset([e for e in EVENTS
                               if e not in TEST_EVENTS + VAL_EVENTS]),
              'val': subset(VAL_EVENTS), 'test': subset(TEST_EVENTS)}
    summary = {'labels': labels, 'test_events': TEST_EVENTS, 'val_events': VAL_EVENTS}
    for name, df in splits.items():
        ids = df['tweet_id'].astype(str).tolist()
        (SPLITS / f'{name}_ids.json').write_text(json.dumps(ids))
        dist = df['class_label'].value_counts().to_dict()
        summary[name] = {'n': len(df), 'class_distribution': dist,
                         'events': sorted(df['event'].unique().tolist())}
        print(f'{name}: n={len(df)} events={summary[name]["events"]}', flush=True)
    (SPLITS / 'split_summary.json').write_text(json.dumps(summary, indent=1))
    print('saved splits + split_summary.json', flush=True)


if __name__ == '__main__':
    main()
