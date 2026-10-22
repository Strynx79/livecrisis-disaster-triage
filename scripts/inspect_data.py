"""One-off probe: per-event sizes + class distribution for QCRI/HumAID-events."""
import json
from collections import Counter
from datasets import load_dataset

EVENTS = ['hurricane_florence_2018', 'kaikoura_earthquake_2016', 'kerala_floods_2018',
          'hurricane_harvey_2017', 'hurricane_maria_2017', 'midwestern_us_floods_2019',
          'puebla_mexico_earthquake_2017', 'maryland_floods_2018', 'hurricane_irma_2017',
          'ecuador_earthquake_2016', 'cyclone_idai_2019', 'canada_wildfires_2016',
          'italy_earthquake_aug_2016', 'greece_wildfires_2018', 'hurricane_dorian_2019',
          'california_wildfires_2018', 'pakistan_earthquake_2019', 'hurricane_matthew_2016',
          'srilanka_floods_2017']

report = {}
total = 0
for ev in EVENTS:
    ds = load_dataset('QCRI/HumAID-events', ev, split='train')
    counts = Counter(ds['class_label'])
    n = len(ds)
    total += n
    report[ev] = {'n': n, 'classes': dict(sorted(counts.items())),
                  'cols': ds.column_names}
    print(f'{ev}: n={n}', flush=True)

print(f'TOTAL: {total}', flush=True)
with open('data/inspect_report.json', 'w') as f:
    json.dump(report, f, indent=1)
print('saved data/inspect_report.json', flush=True)
