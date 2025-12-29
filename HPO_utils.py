import numpy as np
import optuna
from Main import run_experiment

def best_threshold_by_f1(scores: np.ndarray, labels: np.ndarray):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("EVAL must contain both anomalies (1) and normal (0).")

    thr_candidates = np.unique(scores)

    best = None
    for thr in thr_candidates:
        y_pred = (scores >= thr).astype(int)

        tp = np.sum((y_pred == 1) & pos)
        fn = np.sum((y_pred == 0) & pos)
        fp = np.sum((y_pred == 1) & neg)

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        if (best is None) or (f1 > best[0]) or (f1 == best[0] and precision > best[1]):
            best = (f1, precision, recall, thr)

    f1, precision, recall, thr = best
    return float(thr), float(precision), float(recall), float(f1)

def make_objective(data_name: str = 'Linux'):
    def objective(trial: optuna.Trial):
        params = dict(
            data=data_name,
            data_seed=1213,
            num_layers=trial.suggest_int("num_layers", 1, 2),
            aggregation="Mean",
            hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128]),
            lr=trial.suggest_categorical("lr", [1e-4, 1e-3]),
            weight_decay=trial.suggest_categorical("weight_decay", [1e-4, 1e-3]),
            batch=trial.suggest_categorical("batch", [32, 64]),
            device=0,
            hpo=True
        )

        important_epoch_info, epochinfo, *_ = run_experiment(**params)

        best_epoch = max(epochinfo[1:], key=lambda e: e.ap)

        scores = best_epoch.dists.detach().cpu().numpy() if hasattr(best_epoch.dists, "detach") else np.asarray(best_epoch.dists)
        labels = best_epoch.labels.detach().cpu().numpy() if hasattr(best_epoch.labels, "detach") else np.asarray(best_epoch.labels)

        thr, precision, recall, f1 = best_threshold_by_f1(scores, labels)

        # Store extras for inspection
        trial.set_user_attr("thr", thr)
        trial.set_user_attr("f1", f1)
        trial.set_user_attr("recall", recall)
        trial.set_user_attr("precision", precision)

        # Multi-objective: maximize recall and precision
        return recall, precision
    return objective


def run_pareto(n_trials: int = 60, seed: int = 42, dataset_name: str = 'Linux'):
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=seed),
    )
    study.optimize(make_objective(dataset_name), n_trials=n_trials)

    pareto = study.best_trials

    print(f"\nPareto solutions: {len(pareto)}")

    pareto_sorted = sorted(
        pareto,
        key=lambda t: (-t.values[0], t.values[1])
    )[:5]

    for i, t in enumerate(pareto_sorted, 1):
        recall, precision = t.values
        print(f"{i:02d}) recall={recall:.4f}, precision={precision:.4f}, thr={t.user_attrs.get('thr'):.6g}, f1={t.user_attrs.get('f1'):.4f}")
        print(f"    params: {t.params}")

    return study

def pick_solution(study: optuna.Study, min_recall: float = 0.99):
    feasible = [t for t in study.best_trials if t.values[0] >= min_recall]  # values = (recall, precision)
    if not feasible:
        print(f"No Pareto solution reaches recall >= {min_recall}. Try lowering min_recall.")
        return None

    best = max(feasible, key=lambda t: (t.values[1], t.values[0]))  # (precision, recall)
    return best