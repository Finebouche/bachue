"""
Hyperparameters. The three studies of the reservoirs (Optuna, one SQLite database per study): readout_window_study (the
readout: last, mean, tail, ...), learning_rules_study (the learning rules: LMS, NLMS, RLS, EH, and all the monomials of order
2), partial_order2_study (a rule on selected products of order 2); their shared parts in studies. And the choice of
the hyperparameters of the readouts by cross-validation on the train set (ridge, readout_rules, monomial_selection,
monomial_selection_rules, benchmark_rules).
"""
