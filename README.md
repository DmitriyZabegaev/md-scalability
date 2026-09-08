# Parallel Molecular Dynamics: Python/MPI, C++/MPI, and MATLAB/parfor

This repository contains the source code used in the experimental investigation of the scalability of parallel implementations of a molecular dynamics algorithm with a short-range Lennard–Jones potential.

Three implementations of the molecular dynamics algorithm are provided:

- **Python/MPI** — parallel implementation using Python and MPI;
- **C++/MPI** — parallel implementation using C++ and MPI;
- **MATLAB/parfor** — parallel implementation using MATLAB and `parfor`.

The implementations were used to perform a reproducible comparative experiment and to investigate the execution time and scalability of the considered molecular dynamics algorithm.

---

## Scientific Problem

The considered problem is the classical molecular dynamics simulation of interacting particles in three-dimensional space.

The interaction between particles is described by the **Lennard–Jones potential**. To reduce the computational cost of calculating short-range interactions, a truncated and smoothed form of the potential is used.

The equations of motion are integrated using the **Velocity Verlet integration scheme**.

A spatial **cell-list method** is used to identify neighboring particles and reduce the number of pairwise interaction calculations.

The main purpose of the computational experiment is to compare different parallel implementations of the same molecular dynamics algorithm and to investigate their scalability with increasing numbers of parallel processes or workers.

---

## Implementations

| Implementation | Language | Parallelization |
|---|---|---|
| Python/MPI | Python | MPI (`mpi4py`) |
| C++/MPI | C++ | MPI |
| MATLAB/parfor | MATLAB | `parfor` |

The three implementations use the same general molecular dynamics approach, while the parallelization is performed using different technologies.

### Python/MPI

The Python implementation uses the `mpi4py` package to provide MPI-based parallel execution.

The implementation is intended for distributed-memory parallel computing and can be executed using an MPI launcher.

### C++/MPI

The C++ implementation uses MPI for distributed-memory parallelization.

The program is compiled with an MPI-enabled C++ compiler and executed using an MPI launcher.

### MATLAB/parfor

The MATLAB implementation uses the `parfor` construct provided by the MATLAB Parallel Computing Toolbox.

This implementation is intended for shared-memory parallel execution using MATLAB workers.

---

## Repository Structure

```text
molecular-dynamics-parallel/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
│
├── python/
│   └── main_src.py
│
├── cpp/
│   └── main_src.cpp
│
├── matlab/
│   └── ...
│
└── paper/
    └── ...
