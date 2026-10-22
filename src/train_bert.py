"""Overnight (conditional): bert-base-uncased, same protocol as DistilBERT.

3 epochs, lr 2e-5, max_len 128, fp16, effective batch 32 via 8x4
(fallback 4x8 on OOM). Reuses train_distilbert.main() via module TAG override.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_distilbert as T

T.MODEL_NAME = 'bert-base-uncased'
T.TAG = 'bert_base'
T.EPOCHS = 3
T.ATTEMPTS = [(8, 4), (4, 8)]

if __name__ == '__main__':
    T.main()
