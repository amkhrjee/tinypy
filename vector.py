import math


class Vector:
    """
    A class to represent a vector.

    Attributes:
    ----------
    data : list
        List to store vector elements.
    dim : int
        Dimension of the vector.
    """

    def __init__(self, data):
        """
        Constructs all the necessary attributes for the vector object.

        Parameters:
        ----------
        data : list
            List of initial values for the vector elements.
        """
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

    def dot(self, rhs) -> float:
        """
        Computes the dot product of the vector with another vector.

        Parameters:
            Vector: The vector to dot with.

        Returns:
            float: The dot product of the two vectors.

        Raises:
            ValueError: If rhs is not a Vector or if the dimensions do not match.
        """
        if not isinstance(rhs, Vector):
            raise ValueError("Can only dot with a vector")
        if rhs.dim != self.dim:
            raise ValueError(
                f"Expected {self.dim}-dimensional vector, received {rhs.dim}-dimensional vector instead"
            )
        return sum((self * rhs).data)

    def norm(self):
        """
        Computes the Euclidean norm (magnitude) of the vector.

        Returns:
            float: The Euclidean norm of the vector.
        """
        return math.sqrt(sum([x**2 for x in self.data]))

    def normalize(self):
        """
        Normalize the vector.

        This method scales the vector to have a norm (magnitude) of 1,
        effectively converting it to a unit vector.

        Returns:
            Vector: A new vector that is the normalized version of the original vector.
        """
        return self * (1 / self.norm())

    def _normalize(self):
        norm = self.norm()
        self.data = [x / norm for x in self.data]

    def __str__(self) -> str:
        final = f"{self.dim}-dim Vector at {hex(id(self))}: <"
        for data in self.data:
            final += f"{data: .2f}  "
        final += ">"
        return final
