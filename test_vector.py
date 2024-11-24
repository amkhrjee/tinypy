import random

import numpy as np

from vector import Vector


class TestVector:
    def test_dot(self):
        dim = random.randint(1, 10)
        x = Vector([float(random.randint(-20, 20)) for _ in range(dim)])
        y = Vector([float(random.randint(-20, 20)) for _ in range(dim)])
        assert x.dot(y) == np.array(x.data).dot(np.array(y.data))

    def test_norm(self):
        dim = random.randint(1, 10)
        x = Vector([float(random.randint(-20, 20)) for _ in range(dim)])
        assert abs(np.linalg.norm(np.array(x.data)) - x.norm()) < 1e-2
