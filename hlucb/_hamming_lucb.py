from typing import Callable, Generator, Generic, List, Optional, Tuple, TypeVar

import numpy as np

T = TypeVar('T')


class HammingLUCB(Generic[T]):
    @classmethod
    def from_comparator(
        cls,
        items: List[T],
        k: int,
        h: int,
        delta: float,
        comparator: Callable[[T, T], bool],
        seed: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        n = len(items)
        generator = cls.get_generator(n, k, h, delta, seed=seed)
        result = None
        for (i, j), result in generator:
            generator.send(comparator(items[i], items[j]))
        return result

    @classmethod
    def get_generator(
        cls,
        n: int,
        k: int,
        h: int,
        delta: float,
        seed: Optional[int] = None,
    ) -> Generator[Tuple[Tuple[int, int], Tuple[np.ndarray, np.ndarray]], bool, None]:
        if delta <= 0 or delta >= 1:
            raise ValueError(f"Parameter delta ({delta}) must be in range (0, 1)")

        if n < 2:
            raise ValueError(f"Not enough elements in items ({n} < 2)")

        if k + h >= n:
            raise ValueError("Inequality k + h < n does not hold")

        u = np.zeros(n).astype(int)  # Comparison counters
        tau = np.zeros(n)  # Borda scores
        alpha = np.empty(n)  # Confidence bounds
        alpha[:] = np.nan

        rng = np.random.default_rng(seed=seed)

        # Initialization
        for i in range(0, n):
            j = cls._sample(rng, i, n)
            comparison = yield (i, j), (tau, alpha)
            u[i] += 1
            if comparison:
                tau[i] = 1.0

        # Make comparisons unil termination condition is met
        while True:
            # Determine the score order (sorted from highest to lowest score)
            o = np.argsort(tau)[::-1]

            # Confidence bounds
            beta: np.ndarray = np.log(n / delta) + 0.75 * np.log(np.log(n / delta)) + 1.5 * np.log(1 + np.log(u / 2))
            alpha = np.sqrt(beta / (2 * u))

            def score_argmin(xs: np.ndarray, o: np.ndarray, left: int, right: int) -> int:
                return o[np.argmin(xs[o][left:right]) + left]

            # Index of the lowest lower bound in the high-scoring set
            d1 = score_argmin(tau - alpha, o, 0, k - h)

            # Index of the highest upper bound in the low-scoring set
            d2 = score_argmin(-1 * (tau + alpha), o, k + h, n)

            # Index of highest uncertainty in upper half of middle set
            b1 = score_argmin(-1 * alpha, o, k - h, k)
            if alpha[d1] > alpha[b1]:
                b1 = d1

            # Index of highest uncertainty in lower half of middle set
            b2 = score_argmin(-1 * alpha, o, k, k + h)
            if alpha[d2] > alpha[b2]:
                b2 = d2

            # Update scores based on confidence bounds
            for i in [b1, b2]:
                j = cls._sample(rng, i, n)
                comparison = yield (i, j), (tau, alpha)
                u[i] += 1
                tau[i] = tau[i] * (u[i] - 1) / u[i]
                if comparison:
                    tau[i] += 1.0 / u[i]

            # Termination condition
            if tau[d1] - alpha[d1] >= tau[d2] + alpha[d2]:
                break

    @staticmethod
    def _sample(rng: np.random.Generator, i: int, n: int) -> int:
        j = i
        while j == i:
            j = rng.integers(n)
        return j
