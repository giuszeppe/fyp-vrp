"""RL-focused metrics."""


def generalization_gap(train_metric: float, eval_metric: float) -> float:
    return eval_metric - train_metric


def dynamic_degradation(static_metric: float, dynamic_metric: float) -> float:
    return dynamic_metric - static_metric
