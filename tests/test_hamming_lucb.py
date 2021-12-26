import logging
from typing import List, Tuple

import pytest
import numpy as np

from hlucb import HammingLUCB
from hlucb._plotting import plot_scores

logger = logging.getLogger(__name__)


@pytest.fixture(name='params', scope='module')
def fixture_params() -> Tuple[List[int], int, int, float]:
    items = [12, 20, 4, 1, 11, 18, 19, 15, 7, 14, 2, 6, 17, 5, 10, 13, 8, 16, 9, 3]
    k = 10
    h = 3
    delta = 0.99

    return items, k, h, delta


class TestHammingLUCB:
    def test_comparator(self, params: Tuple[List[int], int, int, float]) -> None:
        items, k, h, delta = params
        HammingLUCB.from_comparator(items, k, h, delta, self._compare, seed=42)

    def test_generator(self, params: Tuple[List[int], int, int, float]) -> None:
        items, k, h, delta = params
        n = len(items)
        generator = HammingLUCB.get_generator(n, k, h, delta, seed=42)
        scores = None
        bounds = None
        for (i, j), (scores, bounds) in generator:
            generator.send(self._compare(items[i], items[j]))

        order = np.argsort(scores)[::-1]
        items_arr = np.array(items)
        plot_scores(items_arr[order], scores[order], bounds[order])

    @staticmethod
    def _compare(item_a: int, item_b: int) -> bool:
        return item_a > item_b
