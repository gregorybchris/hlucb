"""Plotting utilities for Hamming LUCB."""
import numpy as np


def plot_scores(labels: np.ndarray, scores: np.ndarray, bounds: np.ndarray) -> None:
    """Print an ASCII representation of Hamming LUCB algorithm output.

    Item scores and their confidence intervals are plotted in ranked order.

    Args:
        labels: Array of labels per item.
        scores: Array of scores produced by Hamming LUCB.
        bounds: Array of confidence interval bounds.
    """
    w = 100
    print(" " + '-' * w)
    for label, score, bound in zip(labels, scores, bounds):
        center = int(w * score)
        left = int(np.clip(w * (score - bound), 0, w - 1))
        right = int(np.clip(w * (score + bound), 0, w - 1))

        print('|', end='')
        for s in range(w):
            if s < left:
                print(' ', end='')
            elif s == left == center:
                print('@', end='')
            elif s == left:
                print('|', end='')
            elif s < center:
                print('-', end='')
            elif s == center:
                print('@', end='')
            elif s < right:
                print('-', end='')
            elif s == right == center:
                print('@', end='')
            elif s == right:
                print('|', end='')
            else:
                print(' ', end='')
        print(f'| {score:.2f}  ±{bound:.2f} || {label}', end='\n')
    print(" " + '-' * w)
