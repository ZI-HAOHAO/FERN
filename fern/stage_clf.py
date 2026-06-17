"""Train-only multi-class stage classifier.

A flagged flow's stage must be PREDICTED from its features by a model trained ONLY on the
train split -never read from the true label. This module trains a LightGBM multiclass
classifier (Normal + the 5 kill-chain stages) on train flows and predicts stages on any
flow set. Used by both pilot_k1 (K1) and retention (K2) reconstruction.
"""
import numpy as np
from stage_mapping import map_attack, STAGE_ORDER, UNKNOWN

STAGE_CLASSES = ["Normal"] + STAGE_ORDER   # index 0 = benign


def true_stage(f, dataset):
    s, _ = map_attack(f["attack"], dataset)
    if s in (None,):
        return "Normal"
    if s == UNKNOWN:
        return "Normal"          # unmapped -> treat as benign for the classifier target
    return s


def stage_targets(flows, dataset):
    idx = {c: i for i, c in enumerate(STAGE_CLASSES)}
    return np.array([idx[true_stage(f, dataset)] for f in flows], dtype=np.int64)


def train_stage_clf(Xtr, flows_tr, dataset):
    import lightgbm as lgb
    ytr = stage_targets(flows_tr, dataset)
    d = lgb.Dataset(Xtr, label=ytr)
    m = lgb.train({"objective": "multiclass", "num_class": len(STAGE_CLASSES),
                   "verbose": -1, "num_leaves": 31, "learning_rate": 0.1},
                  d, num_boost_round=80)
    return m


def predict_stages(model, Xte):
    """Return array of predicted stage strings (len = n test flows)."""
    proba = model.predict(Xte)
    arg = np.argmax(proba, axis=1)
    return np.array([STAGE_CLASSES[i] for i in arg], dtype=object)



