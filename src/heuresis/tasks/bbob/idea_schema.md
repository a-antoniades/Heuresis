Write `idea.md` with this EXACT structure:

## Strategy
(1-2 sentence thesis: what kind of optimizer we're trying and why we expect it
to beat the current best on BBOB)

## Algorithm

### Name
(Concise name: e.g., "CMA-ES with restarts", "(1+1)-ES", "DE/rand/1/bin",
"Nelder-Mead with random restarts")

### Core mechanism
(The generating distribution and update rule in plain language.)

### Adaptation strategy
(How step size / population / covariance / temperature is adapted.
State "none" explicitly if static.)

### Termination / restart
(What triggers a restart; what to do on stagnation; what to do if
BudgetExhausted is raised.)

### Pseudocode
```
# 10-30 lines of numpy-style pseudocode showing the full outer loop.
```

## Implementation Plan
(Numbered steps: what to change in optimizer.py, in what order)

## Pitfalls to Watch
(Known failure modes: numerical stability of updates, tuning sensitivity,
likely failure patterns at budget=1000 FEvals, etc.)
