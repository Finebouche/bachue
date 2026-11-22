"""
Study 3: the partial readouts of order 2, a rule on the state and a few hundred or thousand products of degree 2
chosen before its training (bachue.models.product_selection), instead of all the monomials of "ridge:2", "nlms:2" and
"neh:2" (bachue.hpo.learning_rules_study): one study per rule and selection, so that they are compared on reservoirs optimized
for each of them (results: bachue.performances.partial_order2_evaluation studies, with the same variables). (The exploratory
comparisons of the selections on the reservoirs of "ridge:2": the other commands of
bachue.performances.partial_order2_evaluation.)

Rule and selection of the study: BACHUE_STUDY_RULE=<rule>:<selection> (studies
"next_phrase_calls_[leak_]<rule>-<selection>_<model>_<size>", the ":" of the selection replaced by "-"):
  - rule: "ridge", "rls", "lms", "nlms", or a version of EH ("eh_np_local", "neh", ...), on the standardized state
    and products, in primal form;
  - selection: one of bachue.models.product_selection, e.g. "random_pairs:2080" (2080 random pairs of units),
    "pca:64" (the 2080 products of the 64 first principal components of the states), "scores" (the 435 products of
    the outputs of the linear readout), "squares", "ofr:2080", "leverage:2080", ... The selection is chosen on the
    training rows of each fold.
For instance "ridge:random_pairs:2080", "nlms:pca:64", "eh_np_local:scores".

Search: the parameters of the reservoirs (bachue.hpo.studies.reservoir_parameters, 400 trials by default) and those of
the rule: its normalized learning rate (LMS, NLMS and EH: "readout:rate", log-uniform in RULE_RATES; one training per
fold), the amplitude of the exploration noise of EH ("readout:noise", log-uniform in RULE_NOISES) and the coefficient
alpha of RLS (initial P = I / alpha, "readout:alpha", log-uniform in RULE_ALPHAS, one epoch); the same ranges as the
rules on the states (bachue.hpo.learning_rules_study), so that the two studies are comparable. Objective: the
out-of-fold accuracy in the 3-fold cross-validation on the train set (the next phrase of the songs, readout
"last" -> "last"; or the class of the instances of a classification dataset of hag, BACHUE_DATASET, the last state
of each instance); the selection of products is chosen on the training rows of each fold, and the other
hyperparameters of the rule (number of epochs, up to MAX_EPOCHS; the ridge coefficient of "ridge:<selection>") in
this cross-validation (the cross-validation of the rules, bachue.hpo.learning_rules_study.choose and
choose_on_instances; user attribute "readout_rule" of the trial). Studies on the reservoirs of 500 units, the 4 pure
models, with and without leak, on marron1 and JapaneseVowels.

Results: bachue.performances.partial_order2_evaluation studies (the final evaluation of these studies).

Usage (BACHUE_TASK=next_phrase BACHUE_WITH_CALLS=1 BACHUE_SEARCH_READOUT=0 BACHUE_STUDY_RULE=<rule>:<selection>
[BACHUE_LEAKY_RESERVOIRS=1]):
  python -m bachue.hpo.partial_order2_study <model> 500     (slurm/submit_hpo_rules.sh with RULES="<rule>:<selection> ...")
"""
import sys

from bachue.hpo.learning_rules_study import choose, choose_on_instances
from bachue.hpo.studies import reservoir_parameters, run_study
from bachue.models.readout_rules.training import rule_inputs
from bachue.settings import HPO_RULE, env
from bachue.tasks.next_phrase import BASE_READOUT

# range of the normalized learning rate of the online rules (c / mean ||x||^2 for LMS and EH, stable below 2;
# c / (||x||^2 + 1) for NLMS and neh)
RULE_RATES = {"lms": (1e-3, 0.3), "eh": (1e-2, 1.0), "eh_np": (1e-2, 1.0), "eh_np_local": (1e-2, 1.0),
              "nlms": (1e-3, 1.0), "neh": (1e-3, 0.3)}
# range of the amplitude of the exploration noise of the EH rules (uniform in [-noise, noise] on each output, one-hot
# targets; Hoerzer et al. 2014: 0.5)
RULE_NOISES = {"eh": (0.05, 4.0), "eh_np": (0.05, 4.0), "eh_np_local": (0.05, 4.0), "neh": (0.05, 4.0)}
# range of the coefficient alpha of RLS (initial P = I / alpha; below 1e-8 the recursion loses its precision)
RULE_ALPHAS = {"rls": (1e-8, 1.0)}
# number of epochs of the online rules chosen up to this one (BACHUE_STUDY_MAX_EPOCHS)
MAX_EPOCHS = int(env("BACHUE_STUDY_MAX_EPOCHS", 3000))


def objective_of_selection(label):
    """objective_of of bachue.hpo.studies.run_study for "<rule>:<selection>", on the songs (the data of
    bachue.hpo.studies.next_phrase_data) or on the instances of a classification dataset (instances_data)."""
    rule = rule_inputs(label)[0]

    def objective_of(model, size, *data):
        def objective(trial):
            params = reservoir_parameters(trial, model, data[-1])
            rate = trial.suggest_float("readout:rate", *RULE_RATES[rule], log=True) if rule in RULE_RATES else None
            noise = trial.suggest_float("readout:noise", *RULE_NOISES[rule], log=True) if rule in RULE_NOISES else None
            alpha = trial.suggest_float("readout:alpha", *RULE_ALPHAS[rule], log=True) if rule in RULE_ALPHAS else None
            if len(data) == 2:  # (the instances of a classification dataset)
                chosen, _ = choose_on_instances(model, params, size, data[0], [label], rate, MAX_EPOCHS, noise, alpha)
            else:  # (the songs)
                songs, train_songs, frame, classes, _ = data
                chosen, _ = choose(model, params, size, songs, train_songs, frame, classes, BASE_READOUT, [label], rate,
                                   MAX_EPOCHS, noise, alpha)
            hyper, accuracy = chosen[label]
            trial.set_user_attr("readout_rule", hyper)
            return accuracy
        return objective
    return objective_of


if __name__ == "__main__":
    if not HPO_RULE or rule_inputs(HPO_RULE)[2] is None:
        raise ValueError(f"BACHUE_STUDY_RULE={HPO_RULE}: a rule and a selection, <rule>:<selection> (all the "
                         "monomials, <rule>:2, and the rules on the states: bachue.hpo.learning_rules_study)")
    run_study(sys.argv[1], int(sys.argv[2]), objective_of_selection(HPO_RULE),
              lambda trial: f"readout {trial.user_attrs['readout_rule']}", tasks=("next_phrase", "classification"))
