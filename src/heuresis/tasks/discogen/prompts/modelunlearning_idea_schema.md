Write `idea.md` with this exact structure:

## Strategy
(1-2 sentence thesis: which mechanism, why we expect it to shift every metric in the right direction.)

## Modification

### What to change
(Exact functions in `discovered/loss.py`, exact code changes. Include any new helper methods, state variables, or hyperparameters.)

### Why this should work
(Technical reasoning: how this loss formulation should improve every metric. Cite prior work only when it directly supports the mechanism.)

## Executor Freedom
(Only narrow tuning choices the executor may make; everything else is fixed. Include starting values and allowed ranges.)

## Implementation Plan
(Numbered steps for editing `discovered/loss.py`.)

## Pitfalls to Watch
(Known failure modes, numerical stability issues, interface violations, and per-metric collapse warning signs.)

Rules:
- Be specific: exact function names and exact changes to `discovered/loss.py`.
- Pick one substantive mechanism per idea; do not stack unrelated changes.
- Keep the change confined to `discovered/loss.py`.
- Do not change the `CustomUnlearnTrainer` class name.
- Do not change the `compute_loss` signature.
