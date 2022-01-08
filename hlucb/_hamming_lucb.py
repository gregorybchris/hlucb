"""Hamming LUCB implementation."""
from typing import Callable, Generator, Generic, List, Optional, Tuple, TypeVar

import numpy as np

T = TypeVar('T')


class HammingLUCB(Generic[T]):
    """Container class for Hamming LUCB implementations.

    This class contains utilities for approximately ranking a set of items.
    """
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
        """Run the Hamming LUCB algorithm with a comparator function.

        The comparator should return True when the first parameter is greater
        than the second, otherwise False.

        Args:
            items: List of items to rank.
            k: Number of items to extract as high-ranking items.
            h: Half the margin between high-ranking and low-ranking items.
            delta: Confidence parameter for the probability of achieving
                h-Hamming accuracy. This should be a float in the range (0, 1).
            comparator: Function that accepts two items and returns a boolean
                whether the first is greater than the second.
            seed: Seed for random generator.

        Returns:
            A tuple of two numpy arrays, the first contains the item scores and
                the second contains the confidence intervals of the scores.
        """
        n = len(items)
        generator = cls.get_generator(n, k, h, delta, seed=seed)
        result = None
        for (i, j), result in generator:
            generator.send(comparator(items[i], items[j]))
        return result

    @staticmethod
    def _score_argm(xs: np.ndarray, o: np.ndarray, left: int, right: int, minmax: str = 'max') -> int:
        """Argmin/max for a reordered subarray.

        Get the index of the smallest item within a range of an array that has
        been reordered based on a supplied ordering.

        Args:
            xs: Items in which to search for the smallest value.
            o: Ordering for the items of xs.
            left: Left index of subarray in which to search.
            right: Right index of subarray in which to search.
            minmax: String 'min' or 'max'.

        Returns:
            Integer index of the smallest value in the reordered subarray.

        Raises:
            ValueError: If `delta` outside the range `(0, 1)`.
            ValueError: If there are not at least two elements to rank.
            ValueError: If `k + h >= n`.
        """
        if right <= left:
            raise ValueError("Invalid bounds on argmin")

        argm = np.argmax if minmax == 'max' else np.argmin
        return o[argm(xs[o][left:right]) + left]

    @classmethod
    def get_generator(
        cls,
        n: int,
        k: int,
        h: int,
        delta: float,
        seed: Optional[int] = None,
    ) -> Generator[Tuple[Tuple[int, int], Tuple[np.ndarray, np.ndarray]], bool, None]:
        """Get a generator object that can be used to run the Hamming LUCB algorithm.

        Args:
            n: Total number of items.
            k: Number of items to extract as high-ranking items.
            h: Half the margin between high-ranking and low-ranking items.
            delta: Confidence parameter for the probability of achieving
                h-Hamming accuracy.
            seed: Seed for random generator.

        Yields:
            A tuple of two numpy arrays, the first contains the item scores and
                the second contains the confidence intervals of the scores at the
                current iteration.

        Raises:
            ValueError: If `delta` outside the range `(0, 1)`.
            ValueError: If there are not at least two elements to rank.
            ValueError: If `k + h >= n`.
        """
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
            j = cls._sample(rng, n, i)
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

            # Index of the lowest lower bound in the high-scoring set
            d1 = cls._score_argm(tau - alpha, o, 0, k - h, minmax='min')

            # Index of the highest upper bound in the low-scoring set
            d2 = cls._score_argm((tau + alpha), o, k + h, n, minmax='max')

            # Index of highest uncertainty in upper half of middle set
            b1 = cls._score_argm(alpha, o, k - h, k, minmax='max')
            if alpha[d1] > alpha[b1]:
                b1 = d1

            # Index of highest uncertainty in lower half of middle set
            b2 = cls._score_argm(alpha, o, k, k + h, minmax='max')
            if alpha[d2] > alpha[b2]:
                b2 = d2

            # Update scores based on confidence bounds
            for i in [b1, b2]:
                j = cls._sample(rng, n, i)
                comparison = yield (i, j), (tau, alpha)
                u[i] += 1
                tau[i] = tau[i] * (u[i] - 1) / u[i]
                if comparison:
                    tau[i] += 1.0 / u[i]

            # Termination condition
            if tau[d1] - alpha[d1] >= tau[d2] + alpha[d2]:
                break

    @staticmethod
    def _sample(rng: np.random.Generator, n: int, i: int) -> int:
        """Pull random samples j in the range [0, n) such that j != i.

        Args:
            rng: Numpy random number generator.
            n: Size of range from which to pull random samples.
            i: Disallowed index in range [0, n).

        Returns:
            Integer index of the smallest value in the reordered subarray.
        """
        j = i
        while j == i:
            j = rng.integers(n)
        return j
