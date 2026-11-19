"""
Study 3: the partial readouts of order 2, a rule on the state and a few hundred or thousand products of degree 2
chosen before its training (bachue.models.product_selection), instead of all the monomials of "ridge:2", "nlms:2" and
"neh:2" (bachue.hpo.learning_rules_study): one study per rule and selection, so that they are compared on reservoirs optimized
for each of them (results: bachue.performances.learning_rules_evaluation with the same variables). (The exploratory comparisons of
the selections on the reservoirs of "ridge:2": bachue.performances.partial_order2_evaluation and monomial_selection_rules.)

Rule and selection of the study: HAG_CANARY_HPO_RULE=<rule>:<selection> (studies
"next_phrase_calls_[leak_]<rule>-<selection>_<model>_<size>", the ":" of the selection replaced by "-"):
  - rule: "ridge", "rls", "lms", "nlms", or a version of EH ("eh_np_local", "neh", ...), on the standardized state
    and products, in primal form;
  - selection: one of bachue.models.product_selection, e.g. "random_pairs:2080" (2080 random pairs of units),
    "pca:64" (the 2080 products of the 64 first principal components of the states), "scores" (the 435 products of
    the outputs of the linear readout), "squares", "ofr:2080", "leverage:2080", ... The selection is chosen on the
    training rows of each fold.
For instance "ridge:random_pairs:2080", "nlms:pca:64", "eh_np_local:scores".

Search: the parameters of the reservoirs (bachue.hpo.studies.reservoir_parameters) and those of the rule, as in
bachue.hpo.learning_rules_study (rate, noise, alpha: the same ranges); objective: the out-of-fold accuracy of the next phrase,
the other hyperparameters of the rule chosen in the cross-validation (user attribute "readout_rule" of the trial).
Studies on the reservoirs of 500 units, the 4 pure models.

Usage (HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 HAG_CANARY_TAIL_HPO=0 HAG_CANARY_HPO_RULE=<rule>:<selection>
[HAG_CANARY_LEAK=1]):
  python -m bachue.hpo.partial_order2_study <model> 500     (slurm/submit_hpo_rules.sh with RULES="<rule>:<selection> ...")
"""
import sys

from bachue.hpo.learning_rules_study import objective_of_rule
from bachue.hpo.studies import run_study
from bachue.models.readout_rules.training import rule_inputs
from bachue.settings import HPO_RULE

if __name__ == "__main__":
    if not HPO_RULE or rule_inputs(HPO_RULE)[2] is None:
        raise ValueError(f"HAG_CANARY_HPO_RULE={HPO_RULE}: a rule and a selection, <rule>:<selection> (all the "
                         "monomials, <rule>:2, and the rules on the states: bachue.hpo.learning_rules_study)")
    run_study(sys.argv[1], int(sys.argv[2]), objective_of_rule(HPO_RULE),
              lambda trial: f"readout {trial.user_attrs['readout_rule']}", tasks=("next_phrase", "classification"))
