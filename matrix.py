import copy
import math

from vector import Vector


class Matrix:
    def __init__(self, n_rows: int = 1, n_cols: int = 1, data=[0.0]):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_elements = n_rows * n_cols
        self.data = [0.0] * self.n_elements
        if len(data) > self.n_elements:
            raise ValueError(
                "Length of array must be less than or equal to the maximum possible number of elements."
            )
        for i, e in enumerate(data):
            self.data[i] = e

    def _sub2ind(self, row: int, col: int):
        return (row * self.n_cols) + col

    def _row(self, idx):
        row_start_idx = idx * self.n_cols
        return self.data[row_start_idx : row_start_idx + self.n_cols]

    def _is_square(self) -> bool:
        return self.n_cols == self.n_rows

    def _set2identity(self):
        if self._is_square():
            for row in range(self.n_rows):
                for col in range(self.n_cols):
                    if row == col:
                        self.set(row, col, 1.0)
                    else:
                        self.set(row, col, 0.0)
            return self
        else:
            raise ValueError(f"Matrix is not square, has shape {self.shape()}")

    def _compare(self, rhs, tolerance: float) -> bool:
        if isinstance(rhs, Matrix):
            if self.shape() == rhs.shape():
                sum_of_squared_diff = 0.0
                for i in range(self.n_elements):
                    sum_of_squared_diff += (self.data[i] - rhs.data[i]) ** 2
                mean = math.sqrt(sum_of_squared_diff / (self.n_elements - 1))
                if mean < tolerance:
                    return True
                else:
                    return False
            else:
                return False
        else:
            raise ValueError(f"Expected type Matrix for rhs, but received {type(rhs)}")

    def _close_enough(left: float, right: float) -> bool:
        return abs(left - right) < 1e-9

    def _separate(self, col: int):
        if col >= self.n_cols or col < 1:
            raise ValueError(
                f"Separation Column index must be between 1 and {self.n_cols - 1}"
            )
        n_cols_1 = col
        n_cols_2 = self.n_cols - col
        matrix_1 = Matrix(self.n_rows, n_cols_1)
        matrix_2 = Matrix(self.n_rows, n_cols_2)
        for curr_row in range(self.n_rows):
            for curr_col in range(self.n_cols):
                if curr_col < col:
                    matrix_1.set(curr_row, curr_col, self.get(curr_row, curr_col))
                else:
                    matrix_2.set(curr_row, curr_col - col, self.get(curr_row, curr_col))
        return matrix_1, matrix_2

    def _join(self, rhs):
        if not isinstance(rhs, Matrix):
            raise ValueError(f"Expected type Matrix but received {type(rhs)}")
        if self.n_rows != rhs.n_rows:
            raise ValueError(f"Expected {self.n_rows} rows, got {rhs.n_rows}")
        old_matrix = copy.deepcopy(self)
        self.n_cols = self.n_cols + rhs.n_cols
        self.n_elements = self.n_rows * self.n_cols
        self.data = [0.0] * self.n_elements
        for row in range(self.n_rows):
            for col in range(old_matrix.n_cols):
                self.set(row, col, old_matrix.get(row, col))
            for col in range(rhs.n_cols):
                self.set(row, old_matrix.n_cols + col, rhs.get(row, col))

    def _swap_rows(self, row_1_idx: int, row_2_idx: int):
        if row_1_idx < 0 or row_1_idx >= self.n_rows:
            raise ValueError(f"Bad value for row_1 must be > 0 and <= {self.n_rows}")
        if row_2_idx < 0 or row_2_idx >= self.n_rows:
            raise ValueError(f"Bad value for row_2 must be > 0 and <= {self.n_rows}")
        temp_row = self._row(row_1_idx)
        self.data[row_1_idx * self.n_cols : row_1_idx * self.n_cols + self.n_cols] = (
            self._row(row_2_idx)
        )
        self.data[row_2_idx * self.n_cols : row_2_idx * self.n_cols + self.n_cols] = (
            temp_row
        )

    def _mult_add(self, mult_row_idx: int, add_to_row_idx: int, factor: float):
        if mult_row_idx < 0 or mult_row_idx >= self.n_rows:
            raise ValueError(f"Bad value for row_1 must be > 0 and <= {self.n_rows}")
        if add_to_row_idx < 0 or add_to_row_idx >= self.n_rows:
            raise ValueError(f"Bad value for row_2 must be > 0 and <= {self.n_rows}")
        for col in range(self.n_cols):
            self.set(
                add_to_row_idx,
                col,
                self.get(add_to_row_idx, col) + factor * self.get(mult_row_idx, col),
            )

    def _find_row_with_max_element(self, col_idx: int, starting_from_row_idx: int = 1):
        if col_idx < 0 or col_idx >= self.n_cols:
            raise ValueError(f"col has to be between 1 and {self.n_cols}")
        if starting_from_row_idx < 0 or col_idx >= self.n_rows:
            raise ValueError(f"starting_from_row has to be between 1 and {self.n_cols}")
        start_row_idx = starting_from_row_idx
        col_idx = col_idx
        max_row_idx = start_row_idx
        max_val = self.get(max_row_idx, col_idx)
        for row in range(start_row_idx + 1, self.n_rows):
            if self.get(row, col_idx) > max_val:
                max_row_idx = row
                max_val = self.get(row, col_idx)
        return max_row_idx

    def _mult_row(self, row_idx: int, factor: float):
        if row_idx < 0 or row_idx > self.n_rows:
            raise ValueError(f"Row must be between 1 and {self.n_cols}")
        for col in range(self.n_cols):
            self.set(row_idx, col, factor * self.get(row_idx, col))

    def resize(self, n_rows: int, n_cols: int):
        if self.n_elements > n_rows * n_cols:
            raise ValueError(
                "Total number of elements less than original. This will cause data loss."
            )
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_elements = n_rows * n_cols
        old_data = self.data
        self.data = [0] * self.n_elements
        for i, e in enumerate(old_data):
            self.data[i] = e

    def get(self, row_idx: int, col_idx: int):
        return self.data[self._sub2ind(row_idx, col_idx)]

    def set(self, row_idx: int, col_idx: int, value) -> bool:
        self.data[self._sub2ind(row_idx, col_idx)] = value
        return True

    def inv(self):
        if not self._is_square():
            raise ValueError("Matrix is not invertible")
        identity_matrix = Matrix(self.n_rows, self.n_cols)._set2identity()
        if self == identity_matrix:
            return self
        # Gauss-Jordan algorithm
        # as described in: https://www.statlect.com/matrix-algebra/Gauss-Jordan-elimination
        self._join(identity_matrix)
        for row in range(self.n_rows):
            for col in range(int(self.n_cols / 2)):
                is_current_col_all_zero = True
                for r in range(row, self.n_rows):
                    if not Matrix._close_enough(self.get(r, col), 0.0):
                        is_current_col_all_zero = False
                        break
                if is_current_col_all_zero:
                    continue
                else:
                    max_row = self._find_row_with_max_element(col, row)
                    if not max_row == row:
                        self._swap_rows(max_row, row)
                    self._mult_row(row, 1 / self.get(row, col))
                    for upper_row in range(row):
                        self._mult_add(row, upper_row, -self.get(upper_row, col))
                    for lower_row in range(row + 1, self.n_rows):
                        self._mult_add(row, lower_row, -self.get(lower_row, col))

        # This algorithm has the quirk that the identity matrix is formed
        # in a mirrored fashion like
        # 0 0 1
        # 0 1 0
        # 1 0 0
        # this is a quick hack to fix that

        for row in range(int(self.n_rows / 2)):
            self._swap_rows(row, int(self.n_rows) - row - 1)
        left, right = self._separate(int(self.n_cols / 2))

        if left == identity_matrix:
            return right
        else:
            return None

    def _submatrix(self, col_idx: int):
        if col_idx < 0 or col_idx >= self.n_cols:
            raise ValueError(f"col_idx must be > 0 and < {self.n_cols}")
        submatrix = Matrix(self.n_rows - 1, self.n_cols - 1)
        for row in range(1, self.n_rows):
            submatrix_col_idx = 0
            for col in range(self.n_cols):
                if col != col_idx:
                    submatrix.set(row - 1, submatrix_col_idx, self.get(row, col))
                    submatrix_col_idx += 1
        return submatrix

    def det(self):
        if not self._is_square():
            raise ValueError("Cannot compute determinant for non-square matrix")
        if self.shape() == (2, 2):
            return self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
        else:
            sign = 1
            cum_sum = 0.0
            for col in range(self.n_cols):
                submatrix = self._submatrix(col)
                determinant = submatrix.det()
                cum_sum += self.get(0, col) * determinant * sign
                sign = -sign
            return cum_sum

    def solve(self, constants):
        if not isinstance(constants, Vector):
            raise ValueError("Argument must be a vector")
        if constants.dim != self.n_cols:
            raise ValueError("Constants vector has incompatible dimensionality")
        constants = Matrix(constants.dim, 1, constants.data)
        self._join(constants)
        # Gauss-Jordan algorithm
        for row in range(self.n_rows):
            for col in range(int(self.n_cols - 1)):
                is_current_col_all_zero = True
                for r in range(row, self.n_rows):
                    if not Matrix._close_enough(self.get(r, col), 0.0):
                        is_current_col_all_zero = False
                        break
                if is_current_col_all_zero:
                    continue
                else:
                    max_row = self._find_row_with_max_element(col, row)
                    if not max_row == row:
                        self._swap_rows(max_row, row)
                    self._mult_row(row, 1 / self.get(row, col))
                    for upper_row in range(row):
                        self._mult_add(row, upper_row, -self.get(upper_row, col))
                    for lower_row in range(row + 1, self.n_rows):
                        self._mult_add(row, lower_row, -self.get(lower_row, col))
        for row in range(int(self.n_rows - 1)):
            self._swap_rows(row, int(self.n_rows) - row - 1)
        left, right = self._separate(int(self.n_cols - 1))
        if left == Matrix(self.n_rows, self.n_cols - 1)._set2identity():
            return Vector(right.data)
        else:
            return None

    def __getitem__(self, key):
        return self._row(key)

    def shape(self):
        return (self.n_rows, self.n_cols)

    def __add__(self, rhs):
        if isinstance(rhs, Matrix):
            if self.shape() == rhs.shape():
                new_data = [0] * self.n_elements
                for i, _ in enumerate(self.data):
                    new_data[i] = self.data[i] + rhs.data[i]
                return Matrix(self.n_rows, self.n_cols, new_data)
            else:
                raise (
                    ValueError(
                        f"The shape lhs is {self.shape()} but the shape of the rhs is {rhs.shape()}."
                    )
                )
        elif isinstance(rhs, (int, float)):
            new_data = [0] * self.n_elements
            for i, _ in enumerate(self.data):
                new_data[i] = self.data[i] + rhs
            return Matrix(self.n_rows, self.n_cols, new_data)
        else:
            raise ValueError("Rhs must be a Matrix or a scalar.")

    def __sub__(self, rhs):
        if isinstance(rhs, Matrix):
            if self.shape() == rhs.shape():
                new_data = [0] * self.n_elements
                for i, _ in enumerate(self.data):
                    new_data[i] = self.data[i] - rhs.data[i]
                return Matrix(self.n_rows, self.n_cols, new_data)
            else:
                raise (
                    ValueError(
                        f"The shape lhs is {self.shape()} but the shape of the rhs is {rhs.shape()}."
                    )
                )
        elif isinstance(rhs, (int, float)):
            new_data = [0] * self.n_elements
            for i, _ in enumerate(self.data):
                new_data[i] = self.data[i] - rhs
            return Matrix(self.n_rows, self.n_cols, new_data)
        else:
            raise ValueError("Rhs must be a Matrix or a scalar.")

    def __mul__(self, rhs):
        if isinstance(rhs, Matrix):
            if self.n_cols == rhs.n_rows:
                new_matrix = Matrix(self.n_rows, rhs.n_cols)
                for lhs_row_idx in range(self.n_rows):
                    for rhs_col_idx in range(rhs.n_cols):
                        cum_sum = 0
                        for lhs_col_idx in range(self.n_cols):
                            cum_sum += self.get(lhs_row_idx, lhs_col_idx) * rhs.get(
                                lhs_col_idx, rhs_col_idx
                            )
                        new_matrix.set(lhs_row_idx, rhs_col_idx, cum_sum)
                return new_matrix
            else:
                raise (
                    ValueError(
                        f"The shape lhs is {self.shape()} but the shape of the rhs is {rhs.shape()}."
                    )
                )
        elif isinstance(rhs, (int, float)):
            return Matrix(self.n_rows, self.n_cols, [x * rhs for x in self.data])
        elif isinstance(rhs, Vector):
            if rhs.dim != self.n_cols:
                raise ValueError("Vector and Matrix dimensions are incompatible")
            col_matrix = self * Matrix(rhs.dim, 1, rhs.data)
            return Vector(col_matrix.data)
        else:
            raise ValueError("Rhs must be a Matrix or a Vector or a scalar.")

    def __eq__(self, rhs) -> bool:
        if (not isinstance(rhs, Matrix)) or (self.shape() != rhs.shape()):
            return False
        for i in range(self.n_elements):
            if not Matrix._close_enough(self.data[i], rhs.data[i]):
                return False
        return True

    def __str__(self) -> str:
        final = f"{self.n_rows}x{self.n_cols} Matrix at {hex(id(self))}"
        final += "\n-------------------------------------\n"
        for i, e in enumerate(self.data):
            if (i + 1) % self.n_cols == 0:
                final += f"{e :.2f}\n"
            else:
                final += f"{e:.2f}\t"
        return final
