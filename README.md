# Hamming-LUCB

The `hlucb` package is a Python implementation of the paper [Approximate ranking from pairwise comparisons](http://proceedings.mlr.press/v84/heckel18a.html).

> Heckel, R., Simchowitz, M., Ramchandran, K., & Wainwright, M. (2018, March). Approximate ranking from pairwise comparisons. In International Conference on Artificial Intelligence and Statistics (pp. 1057-1066). PMLR.

## Installation

`hlucb` is installable with pip:

```bash
$ pip install hlucb
```

## Usage

### Ranking with a comparator

```python
from hlucb import HammingLUCB

items = [4, 1, 2, 6, 5, 8, 9, 3]
k = 5
h = 2
delta = 0.9

def compare(item_a: int, item_b: int) -> bool:
    return item_a > item_b

scores, bounds = HammingLUCB.from_comparator(items, k, h, delta, compare, seed=42)

print("Scores: ", scores)
print("Bounds: ", bounds)
```

### Ranking with a generator

```python
from hlucb import HammingLUCB

items = [4, 1, 2, 6, 5, 8, 9, 3]
n = len(items)
k = 5
h = 2
delta = 0.9

generator = HammingLUCB.get_generator(n, k, h, delta, seed=42)
scores = None
bounds = None
for (i, j), (scores, bounds) in generator:
    comparison = items[i] > items[j]
    generator.send(comparison)

print("Scores: ", scores)
print("Bounds: ", bounds)
```

## Intuition

The Hamming-LUCB algorithm approximately ranks $n$ items, estimates the score of each item, and provides confidence bounds for each score. The intuition behind the approximate ranking is that it's easier to compare items with very different scores, so it should be possible to separate high-scoring items from low-scoring items with few comparisons and high confidence even if the exact ranking is not discovered.

The sets of high- and low-scoring items are designated $S_1$ and $S_2$ respectively. Hamming-LUCB extracts $S_1$ and $S_2$ such that all items in $S_1$ are expected to have higher scores than all items in $S_2$.

Parameters:

- $n$ - the total number of items
- $k$ - the number of items to extract as high-scoring items
- $h$ - half the margin between $S_1$ and $S_2$
- $\delta$ - confidence parameter for the probability of achieving $h$-Hamming accuracy

Definitions:

- The Hamming distance between two sets: $D_H(S_1, S_2) = \lvert (S_1 \cup S_2) \setminus (S_1 \cap S_2) \rvert$
- A ranking $\hat{S_1}$, $\hat{S_2}$ is $h$-Hamming accurate if: $D_H(\hat{S_l}, S_l) \leq 2h$ for
  - $\lvert \hat{S_l} \rvert = \lvert S_l \rvert$
  - $l \in \{1, 2\}$
- A ranking algorithm is $(h, \delta)$-accurate if the ranking returned is $h$-Hamming accurate with probability at least $1 - \delta$.
