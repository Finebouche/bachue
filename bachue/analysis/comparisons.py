"""
Paired comparisons by seed of the results of the evaluations (bachue.performances): mean difference of a metric, its
bootstrap 95 % confidence interval, the p-value of the Wilcoxon signed-rank test and the number of seeds where the
first is better. compare: each condition with the reference one, the HAG models with their projections and the
hybrids with their parts; compare_readouts: each readout with "last" -> "last"; compare_rules: each rule with ridge.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from bachue.models.reservoirs import ESN_MODELS, HAG_MODELS


def compare(results, metric, reference, nb_bootstrap=10000):
    """Paired comparisons by seed of the metric (condition A - condition B), for each size: each condition with the
    reference condition, the HAG models with their projections and the hybrids with their parts."""
    rows = []
    rng = np.random.default_rng(0)
    mfcc = results[results.condition == "mfcc"]
    for size, by_size in results[results.condition != "mfcc"].groupby("network_size"):
        by_size = pd.concat([by_size, mfcc])
        f1 = by_size.pivot_table(index="seed", columns="condition", values=metric)
        pairs = [(c, reference) for c in f1.columns if c != reference]
        pairs += [(hag, f"{hag}_proj") for hag in HAG_MODELS]
        for hag in HAG_MODELS:
            for esn in ESN_MODELS:
                hybrid = f"{hag}+{esn}"
                pairs += [(hybrid, hag), (hybrid, esn)]
        for a, b in dict.fromkeys(pairs):
            if a not in f1.columns or b not in f1.columns:
                continue
            diff = (f1[a] - f1[b]).dropna().to_numpy()
            if len(diff) == 0:
                continue
            means = rng.choice(diff, size=(nb_bootstrap, len(diff))).mean(axis=1)
            p_value = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            rows.append(dict(network_size=size, a=a, b=b, nb_seeds=len(diff), mean_diff=diff.mean(),
                             ci_low=np.percentile(means, 2.5), ci_high=np.percentile(means, 97.5),
                             wilcoxon_p=p_value, a_better=int((diff > 0).sum())))
    return pd.DataFrame(rows)


def filled_tails(results):
    """The results with the tails of the training and of the decision filled (-1 for none; the decision tail of the
    results without this column, before TAIL_SEPARATE, is the tail of the readout)."""
    decision_tail = results["decision_tail"] if "decision_tail" in results else results["tail"]
    return results.assign(tail=results["tail"].fillna(-1), decision_tail=decision_tail.fillna(-1))


def compare_readouts(results, metric="accuracy"):
    """Paired comparisons by seed of the metric of each readout with that of "last" -> "last" (readout - last), for
    each condition and size."""
    keys = ["condition", "network_size", "seed"]
    readouts = filled_tails(results)
    base = readouts[(readouts.points == "last") & (readouts.decision == "last")].set_index(keys)[metric]
    rows = []
    for (condition, size, points, decision, tail, decision_tail), group in readouts.groupby(
            ["condition", "network_size", "points", "decision", "tail", "decision_tail"]):
        if points == decision == "last":
            continue
        diff = (group.set_index(keys)[metric] - base).dropna().to_numpy()
        if len(diff) == 0:
            continue
        rows.append(dict(condition=condition, network_size=size, points=points, decision=decision,
                         tail=None if tail < 0 else int(tail), decision_tail=None if decision_tail < 0 else int(decision_tail),
                         nb_seeds=len(diff), mean=group[metric].mean(),
                         mean_diff=diff.mean(), wilcoxon_p=wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0,
                         a_better=int((diff > 0).sum())))
    return pd.DataFrame(rows)


def compare_rules(results, metric="accuracy"):
    """Paired comparisons by seed of the metric of each rule with that of ridge (rule - ridge), for each condition."""
    keys = ["condition", "network_size", "seed"]
    base = results[results.rule == "ridge"].set_index(keys)[metric]
    rows = []
    for (condition, size, rule), group in results[results.rule != "ridge"].groupby(["condition", "network_size", "rule"]):
        diff = (group.set_index(keys)[metric] - base).dropna().to_numpy()
        if len(diff):
            rows.append(dict(condition=condition, network_size=size, rule=rule, nb_seeds=len(diff),
                             mean=group[metric].mean(), mean_diff=diff.mean(),
                             wilcoxon_p=wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0,
                             a_better=int((diff > 0).sum())))
    return pd.DataFrame(rows)
