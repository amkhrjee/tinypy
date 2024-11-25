# tinypy

A tiny linear algebra library that I wrote to teach myself numerical linear algebra. It does not depend on any third party libraries.

Check out [`examples.ipynb`](https://amkhrjee.github.io/tinypy/) to see what this library can do.

## Features

Supports two data types - `Matrix` and `Vector`.

The `Matrix` type has several useful methods like 

| Method | Function |
|------------------|-----------------|
| `trace()`   | $\text{trace}(A)$   |
| `det()`   | $\text{det}(A)$   |
| `inv()`   | $A^{-1}$   |
| `rank()`   | $\text{rank}(A)$   |
|  `conjugate()` | $\overline{A}$ |
|  `transpose()` | $A^{T}$ |
|  `solve(b)` | Finds $x$ for $Ax=b$ |
|  `qe_decomposition` | Finds $Q$ and $R$ for $A = QR$ |
|  `_hessenberg()` | Finds the upper Hessenberg form of $A$ |
|  `eig()` | Finds the eigenvalues for $A$ |

The library supports the usual basic binary operations between matrices, scalars and vectors.

The `Vector` type supports the following methods:

| Method | Function |
|------------------|-----------------|
| `dot()`   | $\mathbf{a} \cdot \mathbf{b}$   |
| `norm()`   |$\|\mathbf{a}\|$   |
| `normalize()`   | $\dfrac{\mathbf{a}}{\|\mathbf{a}\|}$   |

## Benchmarks for fun



## Notes

While I have thoroughly tested the correctness of the library against `numpy`, it is not meant to be used in any kind of production environment. Just use the standard alternatives.

Not publishing on any package repository for practical reasons.

This project was inspired by the playlist [Linear Algebra in C++](https://www.youtube.com/playlist?list=PL3WoIG-PLjSv9vFx2dg0BqzDZH_6qzF8-).