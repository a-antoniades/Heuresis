Write `idea.md` with this EXACT structure:

## Strategy
(1-2 sentence thesis: what we're trying and why we expect it to improve val_bpb)

## Modification

### What to change
(Exact parameters, exact values, exact code locations in train.py)

### Why this should work
(Technical reasoning: how does this change affect training dynamics,
model capacity, or optimization landscape?)

## Executor Freedom
(ONLY these parameters may be tuned by the executor -- everything else
is FIXED. For each: starting value and allowed range.)

## Implementation Plan
(Numbered steps: what to modify in train.py, in what order)

## Pitfalls to Watch
(Known failure modes from past runs, OOM risks, numerical stability)
