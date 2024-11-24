import random

import numpy as np

from matrix import Matrix
from vector import Vector


class TestMatrix:
    def test_sum(self):
        rows = random.randint(1, 50)
        cols = random.randint(1, 50)
        data_x = [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        data_y = [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        x = Matrix(rows, cols, data_x)
        y = Matrix(rows, cols, data_y)
        z = x + y
        x_np = np.array([x._row(i) for i in range(x.n_rows)])
        y_np = np.array([y._row(i) for i in range(y.n_rows)])
        z_np = x_np + y_np
        z_np_flat = [float(x) for x in z_np.flatten()]
        assert z_np_flat == z.data

    def test_sub(self):
        rows = random.randint(1, 50)
        cols = random.randint(1, 50)
        data_x = [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        data_y = [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        x = Matrix(rows, cols, data_x)
        y = Matrix(rows, cols, data_y)
        z = x - y
        x_np = np.array([x._row(i) for i in range(x.n_rows)])
        y_np = np.array([y._row(i) for i in range(y.n_rows)])
        z_np = x_np - y_np
        z_np_flat = [float(x) for x in z_np.flatten()]
        assert z_np_flat == z.data

    def test_transpose(self):
        rows = random.randint(1, 50)
        cols = random.randint(1, 50)
        x = Matrix(
            rows, cols, [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        )
        assert [
            float(i)
            for i in np.array([[x._row(i) for i in range(x.n_rows)]]).T.flatten()
        ] == x.transpose().data

    def test_matmul(self):
        rows = random.randint(1, 50)
        cols = random.randint(1, 50)
        data_x = [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        data_y = [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        x = Matrix(rows, cols, data_x)
        y = Matrix(cols, rows, data_y)
        z = x * y
        x_np = np.array([x._row(i) for i in range(x.n_rows)])
        y_np = np.array([y._row(i) for i in range(y.n_rows)])
        z_np = x_np @ y_np
        z_np_flat = [float(x) for x in z_np.flatten()]
        assert z_np_flat == z.data

    def test_matrix_scalar_mul(self):
        rows = random.randint(1, 50)
        cols = random.randint(1, 50)
        x = Matrix(
            rows, cols, [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        )
        scalar = random.randint(1, 10)
        assert [
            float(i)
            for i in (
                np.array([[x._row(i) for i in range(x.n_rows)]]) * scalar
            ).flatten()
        ] == (x * scalar).data

    def test_matrix_vec_mul(self):
        rows = random.randint(1, 50)
        cols = random.randint(1, 50)
        x = Matrix(
            rows, cols, [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        )
        vec = Vector([float(random.randint(-20, 20)) for _ in range(cols)])
        assert [
            float(i)
            for i in (
                np.array([[x._row(i) for i in range(x.n_rows)]]) @ np.array(vec.data)
            ).flatten()
        ] == (x * vec).data

    def test_rank(self):
        rows = 4
        cols = 6
        x = Matrix(
            rows, cols, [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        )
        assert (
            np.linalg.matrix_rank(np.array([x._row(i) for i in range(x.n_rows)]))
            == x.rank()
        )

    def test_det(self):
        rows = 4
        cols = rows
        x = Matrix(
            rows, cols, [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        )
        x_det = x.det()
        np_det = float(np.linalg.det(np.array([x._row(i) for i in range(x.n_rows)])))
        assert abs(x_det - np_det) < 1e-2

    def test_inv(self):
        rows = 4
        cols = rows
        x = Matrix(
            rows, cols, [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        )
        np_x = np.array([x._row(i) for i in range(x.n_rows)])
        x_inv = x.inv()
        np_x_inv = np.linalg.inv(np_x)
        np_x_inv_flat = [float(x) for x in np_x_inv.flatten()]
        assert len(np_x_inv_flat) == len(x_inv.data)
        for x, y in zip(np_x_inv_flat, x_inv.data):
            assert abs(x - y) < 1e-2

    def test_trace(self):
        rows = random.randint(1, 50)
        cols = rows
        x = Matrix(
            rows, cols, [float(random.randint(-20, 20)) for _ in range(rows * cols)]
        )
        assert np.trace(np.array([x._row(i) for i in range(x.n_rows)])) == x.trace()

    def test_solve(self):
        # test case taken from: https://www.youtube.com/watch?v=m0rg10KX_sI
        rows = 4
        cols = rows
        x = Matrix(rows, cols, [1, 0, 1, 2, 0, 1, -2, 0, 1, 2, -1, 0, 2, 1, 3, -2])
        np_x = np.array([x._row(i) for i in range(x.n_rows)])
        b = Vector([6, -3, -2, 0])
        solutions = x.solve(b)
        np_solutions = np.linalg.solve(np_x, np.array(b.data))
        assert len(np_solutions.flatten()) == solutions.dim
        for x, y in zip(np_solutions.flatten(), solutions.data):
            assert abs(x - y) < 1e-2

    def test_eig(self):
        x = Matrix(3, 3, [21, 20, 7, 3, 6, 2, 11, 15, 7])
        eigens = x.eig()
        np_eigens = np.linalg.eig(np.array([x._row(i) for i in range(x.n_rows)]))[0]
        assert len(np_eigens) == eigens.dim
        for x, y in zip(np_eigens, eigens.data):
            assert abs(x - y) < 1e-2
