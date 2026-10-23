from pathlib import Path

import numpy as np
from transformers import EvalPrediction

from switchboard.eval.metrics import safe_roc_auc
from switchboard.train import train as tr


def test_small_ok_probability_is_softmax_of_class_one():
    logits = np.array([[0.0, 0.0], [0.0, np.log(3.0)], [1000.0, 0.0]])
    probs = tr.small_ok_probability(logits)
    assert np.allclose(probs, [0.5, 0.75, 0.0])


def test_safe_roc_auc_handles_a_single_class():
    assert safe_roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])) is None
    assert safe_roc_auc(np.array([0, 1]), np.array([0.2, 0.8])) == 1.0


def test_compute_metrics():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
    m = tr.compute_metrics(EvalPrediction(predictions=logits, label_ids=np.array([0, 1, 0, 1])))
    assert m["accuracy"] == 0.5
    assert m["roc_auc"] == 0.75  # 3 of 4 positive-negative pairs ranked right


def run(lr, auc):
    return tr.RunResult(lr, auc, 0.5, Path("x"))


def test_pick_best_prefers_higher_auc_and_ranks_missing_last():
    assert tr.pick_best([run(1, 0.6), run(2, 0.7), run(3, None)]).learning_rate == 2
    assert tr.pick_best([run(1, None), run(2, 0.1)]).learning_rate == 2
    assert tr.pick_best([run(1, 0.7), run(2, 0.7)]).learning_rate == 1
