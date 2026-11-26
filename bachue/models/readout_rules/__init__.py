"""
Rules which train a readout. Online: least mean squares (lms) and exploratory Hebbian learning (eh) in JAX (the steps of a pass
over the rows in a compiled lax.scan), and the exploratory Hebbian rule as a NumPy reservoirpy node (eh_numpy, the
reference of eh: the same weights); their versions on the monomials of the state, in dual form (lms_dual, eh_dual);
recursive least squares (rls: reservoirpy's node with the bias as the weight of a constant input). Each file holds
its own rule, its loop over the steps and the update of its weights (eh_dual takes the step of the rule from eh, and
the centered kernel of the monomials from lms_dual); the learning rate of a row, shared by all, is in _utils.
Solved at once by ridge regression: the linear ridge readout or the generalized one, for each ridge coefficient
(ridge_readouts); the generalized readout of Ohkubo & Inubushi (2024) on the monomials of the state, in primal or dual
form (generalized_readout, which also computes the monomials and their kernel for lms_dual); and the ridge regression
on a precomputed kernel (kernel_ridge).

Which version is which (the rules of bachue.performances.learning_rules_evaluation), detailed in the docstrings of lms
and eh_numpy:
  - "rls": RLS (the RLS-based FORCE rule of Sussillo & Abbott 2009), the ridge regression after one pass;
  - "lms": LMS (Widrow & Hoff 1960), a constant learning rate (reservoirpy's LMS);
  - "nlms": NLMS (Nagumo & Noda 1967, Albert & Gardner 1967), the learning rate divided by ||x||^2 + 1 at each row;
  - "eh": the exploratory Hebbian rule of Hoerzer, Legenstein & Maass (2014);
  - "eh_np", "eh_np_local", "neh": ours, EH with the noise known (node perturbation, Fiete & Seung 2006), then a
    reward per output, then the NLMS step (closest published: Burms, Caluwaerts & Dambre 2015);
  - "nlms:<n>", "neh:<n>": nlms and neh on the monomials of Ohkubo & Inubushi (2024) in dual form (kernel LMS, Liu,
    Pokharel & Principe 2008, normalized): ours.
"""
