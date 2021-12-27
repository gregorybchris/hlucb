import numpy as np


def plot_scores(labels: np.ndarray, scores: np.ndarray, bounds: np.ndarray) -> None:
    w = 100
    print(" " + '-' * w)
    for label, score, bound in zip(labels, scores, bounds):
        center = int(w * score)
        left = int(np.clip(w * (score - bound), 0, w))
        right = int(np.clip(w * (score + bound), 0, w))

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
