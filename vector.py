import math


class Vector:
    def __init__(self, data=[0.0]) -> None:
        self.data = data
        self.dim = len(data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        if key < 0 or key >= self.dim:
            raise IndexError(f"key must be be between 0 and {self.dim - 1}")
        self.data[key] = val

    def __add__(self, rhs):
        new_vec = Vector([0.0] * self.dim)
        if isinstance(rhs, (float, int)):
            new_vec.data = [x + rhs for x in self.data]
        elif isinstance(rhs, Vector):
            new_vec.data = [x + y for (x, y) in zip(self.data, rhs.data)]
        return new_vec

    def __sub__(self, rhs):
        new_vec = Vector([0.0] * self.dim)
        if isinstance(rhs, (float, int)):
            new_vec.data = [x - rhs for x in self.data]
        elif isinstance(rhs, Vector):
            new_vec.data = [x - y for (x, y) in zip(self.data, rhs.data)]
        return new_vec

    def __mul__(self, rhs):
        new_vec = Vector([0.0] * self.dim)
        if isinstance(rhs, (float, int)):
            new_vec.data = [x * rhs for x in self.data]
        elif isinstance(rhs, Vector):
            new_vec.data = [x * y for (x, y) in zip(self.data, rhs.data)]
        return new_vec

    def dot(self, rhs):
        if not isinstance(rhs, Vector):
            raise ValueError("Can only dot with a vector")
        if rhs.dim != self.dim:
            raise ValueError(
                f"Expected {self.dim}-dimensional vector, received {rhs.dim}-dimensional vector instead"
            )
        return sum((self * rhs).data)

    def norm(self):
        return math.sqrt(sum([x**2 for x in self.data]))

    def normalize(self):
        return self * (1 / self.norm())

    def _normalize(self):
        norm = self.norm()
        self.data = [x / norm for x in self.data]

    def __str__(self) -> str:
        return f"{self.dim}-dim Vector at {hex(id(self))}: {self.data}"
