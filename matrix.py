import copy
import math

from vector import Vector


class Matrix:
    """
    A two-dimensional real or complex matrix.

    Attributes:
    ----------
    n_rows : int
        Number of rows in the matrix.
    n_cols : int
        Number of columns in the matrix.
    n_elements : int
        Total number of elements in the matrix.
    data : list
        List to store matrix elements.
    """

    def __init__(self, n_rows: int = 1, n_cols: int = 1, data=[0.0]):
        """
        Constructs all the necessary attributes for the matrix object.

        Parameters:
        ----------
        n_rows : int
            Number of rows in the matrix.
        n_cols : int
            Number of columns in the matrix.
        data : list
            List of initial values for the matrix elements.
        """
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
        """
        Converts row and column indices to a single index.

        Parameters:
        ----------
        row : int
            Row index.
        col : int
            Column index.

        Returns:
        -------
        int
            Single index corresponding to the row and column.
        """
        return (row * self.n_cols) + col

    def _row(self, idx):
        """
        Retrieves a row from the matrix.

        Parameters:
        ----------
        idx : int
            Index of the row to retrieve.

        Returns:
        -------
        list
            List of elements in the specified row.
        """
        row_start_idx = idx * self.n_cols
        return self.data[row_start_idx : row_start_idx + self.n_cols]

    def _is_square(self) -> bool:
        """
        Checks if the matrix is square.

        Returns:
        -------
        bool
            True if the matrix is square, False otherwise.
        """
        return self.n_cols == self.n_rows

    def _set2identity(self):
        """
        Sets the matrix to an identity matrix if it is square.

        Returns:
        -------
        Matrix
            The matrix set to identity if it is square.

        Raises:
        ------
        ValueError
            If the matrix is not square.
        """
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
        """
        Compares the matrix with another matrix within a given tolerance.

        Parameters:
        ----------
        rhs : Matrix
            The matrix to compare with.
        tolerance : float
            The tolerance within which the matrices are considered equal.

        Returns:
        -------
        bool
            True if the matrices are equal within the given tolerance, False otherwise.

        Raises:
        ------
        ValueError
            If the rhs is not a Matrix.
        """
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
        """
        Checks if two floating-point numbers are close enough to be considered equal.

        Parameters:
        ----------
        left : float
            The first number.
        right : float
            The second number.

        Returns:
        -------
        bool
            True if the numbers are close enough, False otherwise.
        """
        return abs(left - right) < 1e-9

    def _separate(self, col: int):
        """
        Separates the matrix into two matrices by a given column.

        Parameters:
        ----------
        col : int
            The column index to separate the matrix.

        Returns:
        -------
        tuple
            Two matrices separated by the given column.
        """
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
        """
        Joins the current matrix with another matrix horizontally.

        Parameters:
        rhs (Matrix): The matrix to join with the current matrix. Must be an instance of the Matrix class and have the same number of rows as the current matrix.

        Raises:
        ValueError: If `rhs` is not an instance of the Matrix class.
        ValueError: If the number of rows in `rhs` does not match the number of rows in the current matrix.

        Modifies:
        Updates the current matrix to include the columns of `rhs` appended to its own columns.
        """
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
        """
        Swap two rows in the matrix.

        Parameters:
        row_1_idx (int): The index of the first row to swap. Must be within the range [0, self.n_rows).
        row_2_idx (int): The index of the second row to swap. Must be within the range [0, self.n_rows).

        Raises:
        ValueError: If row_1_idx or row_2_idx are out of the valid range.
        """
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
        """
        Multiplies a row by a factor and adds the result to another row.

        Args:
            mult_row_idx (int): Index of the row to be multiplied.
            add_to_row_idx (int): Index of the row to which the result will be added.
            factor (float): The factor by which to multiply the row.

        Raises:
            ValueError: If `mult_row_idx` or `add_to_row_idx` are out of bounds.

        """
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
        """
        Finds the row index with the maximum element in a specified column, starting from a given row index.

        Args:
            col_idx (int): The index of the column to search in.
            starting_from_row_idx (int, optional): The row index to start searching from. Defaults to 1.

        Returns:
            int: The index of the row with the maximum element in the specified column.

        Raises:
            ValueError: If col_idx is out of the valid range (0 to self.n_cols - 1).
            ValueError: If starting_from_row_idx is out of the valid range (0 to self.n_rows - 1).
        """
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
        """
        Multiplies all elements in a specified row by a given factor.

        Args:
            row_idx (int): The index of the row to be multiplied. Must be between 0 and the number of rows - 1.
            factor (float): The factor by which to multiply each element in the row.

        Raises:
            ValueError: If the row index is out of the valid range.
        """
        if row_idx < 0 or row_idx > self.n_rows:
            raise ValueError(f"Row must be between 1 and {self.n_cols}")
        for col in range(self.n_cols):
            self.set(row_idx, col, factor * self.get(row_idx, col))

    def resize(self, n_rows: int, n_cols: int):
        """
        Resize the matrix to the specified number of rows and columns.

        Parameters:
        n_rows (int): The new number of rows for the matrix.
        n_cols (int): The new number of columns for the matrix.

        Raises:
        ValueError: If the total number of elements in the new matrix is less than the original,
                    as this would cause data loss.

        Notes:
        - The method will reinitialize the matrix data with zeros and copy the existing elements
          to the new matrix. If the new matrix is larger, the additional elements will be zeros.
        """
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

    def get(self, row_idx, col_idx):
        """
        Retrieve elements from the matrix based on the specified row and column indices.
        Parameters:
        row_idx (int or range): The row index or range of row indices to retrieve.
        col_idx (int or range): The column index or range of column indices to retrieve.
        Returns:
        Vector or Matrix or element:
            - If row_idx is a range and col_idx is an int, returns a Vector containing the elements from the specified column across the specified rows.
            - If row_idx is an int and col_idx is a range, returns a Vector containing the elements from the specified row across the specified columns.
            - If both row_idx and col_idx are ranges, returns a Matrix containing the elements from the specified block.
            - If both row_idx and col_idx are ints, returns the single element at the specified position.
        """
        if isinstance(row_idx, range) and isinstance(col_idx, int):
            column_vec_data = []
            for row in row_idx:
                column_vec_data.append(self.get(row, col_idx))
            return Vector(column_vec_data)
        elif isinstance(row_idx, int) and isinstance(col_idx, range):
            row_vec_data = []
            for col in col_idx:
                row_vec_data.append(self.get(row_idx, col))
            return Vector(row_vec_data)
        elif isinstance(row_idx, range) and isinstance(col_idx, range):

            def range_len(r: range):
                return (r.stop - r.start - 1) // r.step + 1

            block_data = []
            for row in row_idx:
                for col in col_idx:
                    block_data.append(self.get(row, col))
            return Matrix(range_len(row_idx), range_len(col_idx), block_data)
        else:
            return self.data[self._sub2ind(row_idx, col_idx)]

    def set(self, row_idx: int, col_idx: int, value) -> bool:
        """
        Sets the value at the specified row and column indices in the matrix.

        Args:
            row_idx (int): The row index where the value should be set.
            col_idx (int): The column index where the value should be set.
            value: The value to set at the specified row and column.

        Returns:
            bool: True if the value was successfully set.
        """
        self.data[self._sub2ind(row_idx, col_idx)] = value
        return True

    def inv(self):
        """
        Compute the inverse of the matrix using the Gauss-Jordan elimination algorithm.
        Returns:
            Matrix: The inverse of the matrix if it exists, otherwise None.
        Raises:
            ValueError: If the matrix is not square and thus not invertible.
        Notes:
            - This method modifies the original matrix.
            - The algorithm used is the Gauss-Jordan elimination as described in:
              https://www.statlect.com/matrix-algebra/Gauss-Jordan-elimination
            - The identity matrix is formed in a mirrored fashion during the process,
              and a quick hack is applied to fix this.
        """
        if not self._is_square():
            raise ValueError("Matrix is not invertible")
        identity_matrix = Matrix(self.n_rows, self.n_cols)._set2identity()
        if self == identity_matrix:
            return self
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
        """
        Generate a submatrix by removing the first row and the specified column.

        Args:
            col_idx (int): The index of the column to be removed. Must be between 0 and self.n_cols - 1.

        Returns:
            Matrix: A new matrix with one less row and one less column, excluding the specified column.

        Raises:
            ValueError: If col_idx is less than 0 or greater than or equal to self.n_cols.
        """
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
        """
        Calculate the determinant of the matrix.

        This method computes the determinant of a square matrix. If the matrix is not square,
        it raises a ValueError. For a 2x2 matrix, it uses the direct formula. For larger matrices,
        it uses a recursive approach by expanding along the first row.

        Returns:
            float: The determinant of the matrix.

        Raises:
            ValueError: If the matrix is not square.
        """
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
        """
        Solves the system of linear equations represented by the matrix.

        Args:
            constants (Vector): A vector representing the constants of the system.

        Returns:
            Vector: A vector representing the solution to the system if a unique solution exists.
            None: If no unique solution exists.

        Raises:
            ValueError: If the argument is not a vector.
            ValueError: If the dimensionality of the constants vector is incompatible.
            ValueError: If no unique solution exists for the system.
            ValueError: If no solution exists for the system.

        Notes:
            This method uses the Gauss-Jordan elimination algorithm to solve the system of equations.
        """
        if not isinstance(constants, Vector):
            raise ValueError("Argument must be a vector")
        if constants.dim != self.n_cols:
            raise ValueError("Constants vector has incompatible dimensionality")
        constants = Matrix(constants.dim, 1, constants.data)
        rank_before_aug = self.rank()
        self._join(constants)
        rank_after_aug = self.rank()
        if rank_before_aug == rank_after_aug and rank_before_aug < self.n_rows:
            raise ValueError("No unique solution exists for the system.")
        elif rank_before_aug < rank_after_aug:
            raise ValueError("No solution exists for the system.")
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

    def is_nonzero(self):
        """
        Check if the matrix contains any non-zero elements.

        Returns:
            bool: True if there is at least one non-zero element in the matrix, False otherwise.
        """
        for el in self.data:
            if el != 0:
                return True
        return False

    def rank(self):
        """
        Calculate the rank of the matrix using the Gaussian elimination method.
        The rank of a matrix is the maximum number of linearly independent row or column vectors in the matrix.
        This method uses Gaussian elimination to determine the rank.
        Returns:
            int: The rank of the matrix.
        References:
            https://cp-algorithms.com/linear_algebra/rank-matrix.html
        Notes:
            - If the matrix is a square matrix and its determinant is non-zero, the rank is equal to the number of columns.
            - The method creates a deep copy of the matrix to perform the Gaussian elimination without modifying the original matrix.
            - The method iterates through each column and performs row operations to reduce the matrix to row echelon form.
            - The rank is incremented for each pivot row found during the elimination process.
        """
        if not self.is_nonzero():
            return 0
        if self._is_square():
            if self.det() != 0:
                return self.n_cols
        matrix = copy.deepcopy(self)
        rank = 0
        selected_rows = [False] * matrix.n_rows
        for curr_col in range(matrix.n_cols):
            curr_row = 0
            while curr_row < matrix.n_rows:
                if (
                    not selected_rows[curr_row]
                    and abs(matrix.get(curr_row, curr_col)) > 1e-9
                ):
                    break
                curr_row += 1

            if curr_row != self.n_rows:
                rank += 1
                selected_rows[curr_row] = True
                for next_col in range(curr_col + 1, self.n_cols):
                    matrix.set(
                        curr_row,
                        next_col,
                        matrix.get(curr_row, next_col) / matrix.get(curr_row, curr_col),
                    )
                for r in range(matrix.n_rows):
                    if r != curr_row and abs(matrix.get(r, curr_col)) > 1e-9:
                        for c in range(curr_col + 1, matrix.n_cols):
                            matrix.set(
                                r,
                                c,
                                matrix.get(r, c)
                                - (matrix.get(curr_row, c) * matrix.get(r, curr_col)),
                            )
        return rank

    def transpose(self):
        """
        Transpose the matrix by swapping rows with columns.

        Returns:
            Matrix: A new matrix that is the transpose of the original matrix.
        """
        transposed_matrix = Matrix(self.n_cols, self.n_rows)
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                transposed_matrix.set(col, row, self.get(row, col))
        return transposed_matrix

    def trace(self):
        """
        Calculate the trace of the matrix.

        The trace is defined as the sum of the elements on the main diagonal
        (from the top left to the bottom right) of a square matrix.

        Returns:
            int or float: The trace of the matrix.

        Raises:
            ValueError: If the matrix is not square.
        """
        if not self._is_square():
            raise ValueError("Trace is only defined for a square matrix")
        return sum(
            [
                self.get(row, col)
                for row in range(self.n_rows)
                for col in range(self.n_cols)
                if row == col
            ]
        )

    def _delete_col(self, col_idx: int):
        """
        Deletes a column from the matrix at the specified index and returns a new matrix.

        Args:
            col_idx (int): The index of the column to be deleted.

        Returns:
            Matrix: A new matrix with the specified column removed.

        Raises:
            IndexError: If the column index is out of range.
        """
        new_matrix = Matrix(self.n_rows, self.n_cols - 1)
        for row in range(self.n_rows):
            passed_deleted_col = False
            for col in range(self.n_cols):
                if col != col_idx:
                    if passed_deleted_col:
                        new_matrix.set(row, col - 1, self.get(row, col))
                    else:
                        new_matrix.set(row, col, self.get(row, col))
                else:
                    passed_deleted_col = True
        return new_matrix

    def _delete_row(self, row_idx):
        """
        Deletes a row from the matrix and returns a new matrix with the row removed.

        Args:
            row_idx (int): The index of the row to be deleted.

        Returns:
            Matrix: A new matrix with the specified row removed.

        Raises:
            IndexError: If the row_idx is out of range.
        """
        new_matrix = Matrix(self.n_rows - 1, self.n_cols)
        passed_deleted_row = False
        for row in range(self.n_rows):
            if row != row_idx:
                for col in range(self.n_cols):
                    if passed_deleted_row:
                        new_matrix.set(row - 1, col, self.get(row, col))
                    else:
                        new_matrix.set(row, col, self.get(row, col))
            else:
                passed_deleted_row = True
        return new_matrix

    def _hessenberg(self):
        """
        Reduces the matrix to upper Hessenberg form using Householder transformations.
        A matrix is in Hessenberg form if all elements below the first subdiagonal are zero.
        This method only works for square matrices.
        Returns:
            Matrix: The upper Hessenberg form of the original matrix.
        References:
            https://www.cs.cornell.edu/~bindel/class/cs6210-f16/lec/2016-10-21.pdf
        Raises:
            ValueError: If the matrix is not square.
        """
        if not self._is_square():
            raise ValueError("Matrix must be square")
        order = self.n_rows
        H = copy.deepcopy(self)
        Q = Matrix(order, order)._set2identity()
        for j in range(order - 1):
            u = H.get(range(j + 1, H.n_rows), j)
            u[0] += math.copysign(u.norm(), u[0])
            v = u.normalize()
            v_matrix = Matrix(v.dim, 1, v.data)

            H_block = H.get(range(j + 1, H.n_rows), range(H.n_cols))
            first_matrix = H_block - (v_matrix * 2) * (v_matrix.transpose() * H_block)
            for row in range(j + 1, H.n_rows):
                for col in range(H.n_cols):
                    H.set(row, col, first_matrix.get(row - j - 1, col))

            H_block = H.get(range(H.n_rows), range(j + 1, H.n_cols))
            second_matrix = H_block - (H_block * (v_matrix * 2)) * v_matrix.transpose()
            for row in range(H.n_rows):
                for col in range(j + 1, H.n_cols):
                    H.set(row, col, second_matrix.get(row, col - j - 1))

            Q_block = Q.get(range(Q.n_rows), range(j + 1, Q.n_cols))
            third_matrix = Q_block - (Q_block * (v_matrix * 2)) * v_matrix.transpose()
            for row in range(Q.n_rows):
                for col in range(j + 1, Q.n_cols):
                    Q.set(row, col, third_matrix.get(row, col - j - 1))
        return H

    def qr_decomposition(self):
        """
        Perform QR decomposition of a matrix using Householder reflections.
        The QR decomposition decomposes a matrix A into a product A = QR, where Q is an orthogonal matrix
        and R is an upper triangular matrix.
        Returns:
            tuple: A tuple containing two matrices (Q, R) where Q is an orthogonal matrix and R is an upper triangular matrix.
        Raises:
            ValueError: If the matrix is not square.
        Notes:
            This implementation uses Householder reflections to compute the QR decomposition. The algorithm
            iteratively transforms the matrix into an upper triangular form by applying orthogonal transformations.
        References:
            - https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
        """
        Q = Matrix(self.n_rows, self.n_rows)._set2identity()
        R = copy.deepcopy(self)

        for j in range(self.n_cols):
            normx = R.get(range(j, self.n_rows), j).norm()
            s = -math.copysign(1, R.get(j, j))
            u1 = R.get(j, j) - s * normx
            w = R.get(range(j, self.n_rows), j) * (1 / u1)
            w[0] = 1
            w_matrix = Matrix(w.dim, 1, w.data)
            tau = -s * u1 / normx

            R_block = R.get(range(j, R.n_rows), range(R.n_cols))
            R_matrix = R_block - (w_matrix * tau) * (w_matrix.transpose() * R_block)
            for row in range(j, R.n_rows):
                for col in range(R.n_cols):
                    R.set(row, col, R_matrix.get(row - j, col))

            Q_block = Q.get(range(Q.n_rows), range(j, self.n_cols))
            Q_matrix = Q_block - (Q_block * w_matrix) * (w_matrix.transpose() * tau)
            for row in range(Q.n_rows):
                for col in range(j, Q.n_cols):
                    Q.set(row, col, Q_matrix.get(row, col - j))
        return Q, R

    def eig(self, limit=100):
        """
        Compute the eigenvalues of a square non-singular matrix using the QR algorithm.

        Parameters:
        limit (int): The maximum number of iterations for the QR algorithm (default is 100).

        Returns:
        Vector: A vector containing the eigenvalues of the matrix.

        References:
        (2008). Methods for Computing Eigenvalues. In: Numerical Linear Algebra. Texts in Applied Mathematics, vol 55. Springer, New York, NY. https://doi.org/10.1007/978-0-387-68918-0_10

        Raises:
        ValueError: If the matrix is not square or is singular.

        Notes:
        - The function first checks if the matrix is square and non-singular.
        - It then reduces the matrix to Hessenberg form.
        - The QR algorithm is applied iteratively to find the eigenvalues.
        - The process stops when the off-diagonal elements are below a certain tolerance or the iteration limit is reached.
        """
        if not self._is_square() or self.det() == 0:
            raise ValueError(
                "Can only determine eigenvalues for square non-singular matrices"
            )
        n = self.n_rows
        tolerance = 1e-9
        hess = self._hessenberg()
        m = n - 1
        k = 0
        eigenvals = Vector([0.0] * n)
        while k < limit and m > 0:
            if abs(hess.get(m, m - 1)) < tolerance:
                eigenvals[m] = hess.get(m, m)
                hess = hess._delete_col(m)
                hess = hess._delete_row(m)
                m -= 1
            Q, R = hess.qr_decomposition()
            hess = R * Q
            k = k + 1
        eigenvals[0] = hess.get(0, 0)
        return eigenvals

    def conjugate(self):
        """
        Returns the complex conjugate of each element in the matrix.

        If an element is a complex number, its complex conjugate is computed.
        Otherwise, the element is returned as is.

        Returns:
            Matrix: A new matrix with the complex conjugate of each element.
        """
        return Matrix(
            self.n_rows,
            self.n_cols,
            [x.conjugate() if isinstance(x, complex) else x for x in self.data],
        )

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
        final += "\n-----------------------------\n"
        for i, e in enumerate(self.data):
            if (i + 1) % self.n_cols == 0:
                final += f"{e :.2f}\n"
            else:
                final += f"{e:.2f}\t"
        return final
