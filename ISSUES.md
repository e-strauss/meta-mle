# Issues

## MCTS/MCGS widen over near-identical grid siblings

Every grid variant becomes its own child, so progressive widening may spend several
expansions on siblings that differ only in a minor parameter. In full-mcgs, 5 of 9
expansions went to the first draft's five smoothing variants (CV 0.744–0.745).

Treating siblings as one node for widening is not a clean fix: a grid may also choose
between distant alternatives (two model families, two feature encodings), which should
stay separate branches. Open: how to tell near-duplicate siblings from distinct ones
(score spread within fold noise, which grid parameter differs, a planner hint).
