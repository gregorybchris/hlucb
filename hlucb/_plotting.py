import numpy as np


def plot_scores(items: np.ndarray, scores: np.ndarray, bounds: np.ndarray) -> None:
    w = 100
    print("     " + '-' * w)
    for item, score, bound in zip(items, scores, bounds):
        center = int(w * score)
        left = int(np.clip(w * (score - bound), 0, w))
        right = int(np.clip(w * (score + bound), 0, w))

        print(f'{item:3d} |', end='')
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
        print(f'| {score:.2f}  ±{bound:.2f}', end='\n')
    print("     " + '-' * w)
