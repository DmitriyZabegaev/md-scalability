"""Parallel molecular-dynamics simulation using Python and MPI.

The program models argon particles interacting through a smoothed and truncated
Lennard-Jones potential. Particle interactions are accelerated with a cell-list
(neighbor-cell) method, while particle-wise force calculations are distributed
among MPI processes.

The numerical algorithm and communication scheme are intentionally kept close
to the original implementation used for the computational experiments.
"""

import math
import time

import numpy as np
from mpi4py import MPI


# -----------------------------------------------------------------------------
# Physical and numerical constants
# -----------------------------------------------------------------------------

ARGON_MASS = 39.938          # Argon mass, atomic mass units
UNIT_CONVERSION = 9648.533977
EPSILON = 0.01
R_MIN = 3.822
R_ON = 2.0 * R_MIN
R_OFF = 2.5 * R_MIN
FINITE_DIFFERENCE_STEP = 0.001


# -----------------------------------------------------------------------------
# Initial conditions
# -----------------------------------------------------------------------------

def initial_cond(N, Lx, Ly, Lz, r_lim, V):
    """Generate initial particle coordinates and velocity components.

    Particles are placed randomly inside the simulation box. A newly generated
    particle is rejected when it is closer than ``r_lim`` to any previously
    placed particle. Up to ten additional placement attempts are made.

    The initial velocity of every particle has magnitude ``V`` and a random
    direction.
    """
    X = np.zeros(N)
    Y = np.zeros(N)
    Z = np.zeros(N)

    # Place the first particle.
    X[0] = Lx * np.random.rand()
    Y[0] = Ly * np.random.rand()
    Z[0] = Lz * np.random.rand()

    failed_to_place_particles = False

    for i in range(1, N):
        # Generate an initial candidate position.
        X[i] = Lx * np.random.rand()
        Y[i] = Ly * np.random.rand()
        Z[i] = Lz * np.random.rand()

        attempts = 0
        position_is_invalid = True

        while position_is_invalid and attempts <= 10:
            position_is_invalid = False

            for j in range(i):
                distance = np.sqrt(
                    (X[i] - X[j]) ** 2
                    + (Y[i] - Y[j]) ** 2
                    + (Z[i] - Z[j]) ** 2
                )

                if distance < r_lim:
                    position_is_invalid = True
                    break

            if position_is_invalid:
                X[i] = Lx * np.random.rand()
                Y[i] = Ly * np.random.rand()
                Z[i] = Lz * np.random.rand()
                attempts += 1

        if attempts >= 10:
            print(
                f"Failed to place {N} particles. "
                f"Only {i} particles have been placed!"
            )
            failed_to_place_particles = True
            break

    if failed_to_place_particles:
        X = X[:i]
        Y = Y[:i]
        Z = Z[:i]

    # Generate initial velocity components.
    Vx = np.zeros(len(X))
    Vy = np.zeros(len(Y))
    Vz = np.zeros(len(Z))

    for i in range(len(X)):
        phi = 2.0 * np.pi * np.random.rand()
        theta = np.pi * np.random.rand()

        Vx[i] = V * np.cos(phi) * np.sin(theta)
        Vy[i] = V * np.sin(phi) * np.sin(theta)
        Vz[i] = V * np.cos(theta)

    return X, Y, Z, Vx, Vy, Vz


# -----------------------------------------------------------------------------
# Lennard-Jones potential
# -----------------------------------------------------------------------------

def U_lj_cut(r):
    """Evaluate the smoothed and truncated Lennard-Jones potential."""
    if r < R_ON:
        weight = 1.0
    elif R_ON <= r <= R_OFF:
        weight = (
            (R_OFF**2 - r**2) ** 2
            * (R_OFF**2 - 3.0 * R_ON**2 + 2.0 * r**2)
            / (R_OFF**2 - R_ON**2) ** 3
        )
    else:
        weight = 0.0

    return weight * EPSILON * ((R_MIN / r) ** 12 - 2.0 * (R_MIN / r) ** 6)


# -----------------------------------------------------------------------------
# Energy calculation
# -----------------------------------------------------------------------------

def E_tot(X, Y, Z, Vx, Vy, Vz):
    """Calculate the total potential and kinetic energy of the system."""
    # Keep the original energy cutoff used by the simulation.
    R_cut = R_off
    N = len(X)

    potential_energy = 0.0

    for i in range(N):
        for j in range(i):
            distance = np.sqrt(
                (X[i] - X[j]) ** 2
                + (Y[i] - Y[j]) ** 2
                + (Z[i] - Z[j]) ** 2
            )

            if distance < R_cut:
                potential_energy += U_lj_cut(distance)

    kinetic_energy = 0.0
    for i in range(N):
        kinetic_energy += (
            UNIT_CONVERSION
            * ARGON_MASS
            * (Vx[i] ** 2 + Vy[i] ** 2 + Vz[i] ** 2)
            / 2.0
        )

    return potential_energy + kinetic_energy


# -----------------------------------------------------------------------------
# Cell-list construction
# -----------------------------------------------------------------------------

def form_cells(
    X,
    Y,
    Z,
    Lx,
    Ly,
    Lz,
    N_cell_x,
    r_cell_x,
    N_cell_y,
    r_cell_y,
    N_cell_z,
    r_cell_z,
):
    """Assign every particle to its corresponding spatial cell."""
    cell_list = np.empty((N_cell_x, N_cell_y, N_cell_z), dtype=object)

    for i in range(N_cell_x):
        for j in range(N_cell_y):
            for k in range(N_cell_z):
                cell_list[i, j, k] = []

    N_p = len(X)

    for i in range(N_p):
        if 0 < X[i] < Lx:
            nx = min(int(np.ceil(X[i] / r_cell_x)), N_cell_x - 1)
        else:
            nx = 0 if X[i] < 0 else N_cell_x - 1

        if 0 < Y[i] < Ly:
            ny = min(int(np.ceil(Y[i] / r_cell_y)), N_cell_y - 1)
        else:
            ny = 0 if Y[i] < 0 else N_cell_y - 1

        if 0 < Z[i] < Lz:
            nz = min(int(np.ceil(Z[i] / r_cell_z)), N_cell_z - 1)
        else:
            nz = 0 if Z[i] < 0 else N_cell_z - 1

        cell_list[nx, ny, nz].append(i)

    return cell_list



def form_Neib(
    Xi,
    Yi,
    Zi,
    Cell_list,
    Lx,
    Ly,
    Lz,
    r_cell_x,
    r_cell_y,
    r_cell_z,
    N_cell_x,
    N_cell_y,
    N_cell_z,
):
    """Return particle indices from the neighboring spatial cells."""
    if 0 < Xi < Lx:
        nx = min(int(np.ceil(Xi / r_cell_x)), N_cell_x - 1)
    else:
        nx = 0 if Xi < 0 else N_cell_x - 1

    if 0 < Yi < Ly:
        ny = min(int(np.ceil(Yi / r_cell_y)), N_cell_y - 1)
    else:
        ny = 0 if Yi < 0 else N_cell_y - 1

    if 0 < Zi < Lz:
        nz = min(int(np.ceil(Zi / r_cell_z)), N_cell_z - 1)
    else:
        nz = 0 if Zi < 0 else N_cell_z - 1

    neighbor_list = []

    for jx in range(-1, 2):
        for jy in range(-1, 2):
            for jz in range(-1, 2):
                if (
                    0 <= nx + jx < N_cell_x
                    and 0 <= ny + jy < N_cell_y
                    and 0 <= nz + jz < N_cell_z
                ):
                    neighbor_list.extend(
                        Cell_list[nx + jx, ny + jy, nz + jz]
                    )

    return neighbor_list


# -----------------------------------------------------------------------------
# Single-particle potential
# -----------------------------------------------------------------------------

def Ui(X, Y, Z, i, Neib_list, R_off):
    """Calculate the potential energy contribution for particle ``i``."""
    potential = 0.0

    for k in Neib_list:
        if i == k:
            continue

        distance = np.sqrt(
            (X[i] - X[k]) ** 2
            + (Y[i] - Y[k]) ** 2
            + (Z[i] - Z[k]) ** 2
        )

        if distance < R_off:
            potential += U_lj_cut(distance)

    return potential



def U_tot(X, Y, Z):
    """Calculate the total potential energy using the original cutoff."""
    R_cut = 2.5
    N = len(X)
    potential_energy = 0.0

    for i in range(N):
        for j in range(i):
            distance = np.sqrt(
                (X[i] - X[j]) ** 2
                + (Y[i] - Y[j]) ** 2
                + (Z[i] - Z[j]) ** 2
            )

            if distance < R_cut:
                potential_energy += U_lj_cut(distance)

    return potential_energy


# -----------------------------------------------------------------------------
# Serial Verlet integrator (reference implementation)
# -----------------------------------------------------------------------------

def quick_Verlet(X, Y, Z, Vx, Vy, Vz, N_steps, dt):
    """Run the serial velocity-Verlet integration."""
    global N_p, Lx, Ly, Lz
    global r_cell_x, r_cell_y, r_cell_z
    global N_cell_x, N_cell_y, N_cell_z, R_off

    Ax_old = np.zeros(N_p)
    Ay_old = np.zeros(N_p)
    Az_old = np.zeros(N_p)

    cell_list = form_cells(
        X,
        Y,
        Z,
        Lx,
        Ly,
        Lz,
        N_cell_x,
        r_cell_x,
        N_cell_y,
        r_cell_y,
        N_cell_z,
        r_cell_z,
    )

    # Calculate accelerations for the initial configuration.
    for i in range(N_p):
        D = np.zeros(N_p)
        D[i] = FINITE_DIFFERENCE_STEP

        neighbor_list = form_Neib(
            X[i],
            Y[i],
            Z[i],
            cell_list,
            Lx,
            Ly,
            Lz,
            r_cell_x,
            r_cell_y,
            r_cell_z,
            N_cell_x,
            N_cell_y,
            N_cell_z,
        )

        Ui_dx = Ui(X + D, Y, Z, i, neighbor_list, R_off)
        Ui_dy = Ui(X, Y + D, Z, i, neighbor_list, R_off)
        Ui_dz = Ui(X, Y, Z + D, i, neighbor_list, R_off)
        Ui_0 = Ui(X, Y, Z, i, neighbor_list, R_off)

        Ax_old[i] = -(Ui_dx - Ui_0) / (
            UNIT_CONVERSION * ARGON_MASS * FINITE_DIFFERENCE_STEP
        )
        Ay_old[i] = -(Ui_dy - Ui_0) / (
            UNIT_CONVERSION * ARGON_MASS * FINITE_DIFFERENCE_STEP
        )
        Az_old[i] = -(Ui_dz - Ui_0) / (
            UNIT_CONVERSION * ARGON_MASS * FINITE_DIFFERENCE_STEP
        )

    Ax = np.zeros(N_p)
    Ay = np.zeros(N_p)
    Az = np.zeros(N_p)

    for _ in range(N_steps):
        # Reflect velocities at the rigid walls.
        for i in range(N_p):
            if X[i] > Lx:
                Vx[i] = -abs(Vx[i])
            if X[i] < 0:
                Vx[i] = abs(Vx[i])

            if Y[i] > Ly:
                Vy[i] = -abs(Vy[i])
            if Y[i] < 0:
                Vy[i] = abs(Vy[i])

            if Z[i] > Lz:
                Vz[i] = -abs(Vz[i])
            if Z[i] < 0:
                Vz[i] = abs(Vz[i])

        # Update coordinates.
        X += Vx * dt + 0.5 * Ax_old * dt**2
        Y += Vy * dt + 0.5 * Ay_old * dt**2
        Z += Vz * dt + 0.5 * Az_old * dt**2

        # Rebuild the cell list.
        cell_list = form_cells(
            X,
            Y,
            Z,
            Lx,
            Ly,
            Lz,
            N_cell_x,
            r_cell_x,
            N_cell_y,
            r_cell_y,
            N_cell_z,
            r_cell_z,
        )

        # Calculate new accelerations.
        for i in range(N_p):
            D = np.zeros(N_p)
            D[i] = FINITE_DIFFERENCE_STEP

            neighbor_list = form_Neib(
                X[i],
                Y[i],
                Z[i],
                cell_list,
                Lx,
                Ly,
                Lz,
                r_cell_x,
                r_cell_y,
                r_cell_z,
                N_cell_x,
                N_cell_y,
                N_cell_z,
            )

            U_dx = Ui(X + D, Y, Z, i, neighbor_list, R_off)
            U_dy = Ui(X, Y + D, Z, i, neighbor_list, R_off)
            U_dz = Ui(X, Y, Z + D, i, neighbor_list, R_off)
            U_0 = Ui(X, Y, Z, i, neighbor_list, R_off)

            Ax[i] = -(U_dx - U_0) / (
                UNIT_CONVERSION * ARGON_MASS * FINITE_DIFFERENCE_STEP
            )
            Ay[i] = -(U_dy - U_0) / (
                UNIT_CONVERSION * ARGON_MASS * FINITE_DIFFERENCE_STEP
            )
            Az[i] = -(U_dz - U_0) / (
                UNIT_CONVERSION * ARGON_MASS * FINITE_DIFFERENCE_STEP
            )

        # Update velocities.
        Vx += 0.5 * (Ax + Ax_old) * dt
        Vy += 0.5 * (Ay + Ay_old) * dt
        Vz += 0.5 * (Az + Az_old) * dt

        Ax_old = Ax.copy()
        Ay_old = Ay.copy()
        Az_old = Az.copy()

    return X, Y, Z, Vx, Vy, Vz


# -----------------------------------------------------------------------------
# MPI-parallel Verlet integrator
# -----------------------------------------------------------------------------

def quick_Verlet_par(
    X,
    Y,
    Z,
    Vx,
    Vy,
    Vz,
    N_steps,
    dt,
    start_elems,
    fin_elems,
    rank,
    numprocs,
    comm,
):
    """Run the velocity-Verlet integration with MPI particle decomposition."""
    global N_p, Lx, Ly, Lz
    global r_cell_x, r_cell_y, r_cell_z
    global N_cell_x, N_cell_y, N_cell_z, R_off

    Ax_old = np.zeros(N_p)
    Ay_old = np.zeros(N_p)
    Az_old = np.zeros(N_p)

    cell_list = form_cells(
        X,
        Y,
        Z,
        Lx,
        Ly,
        Lz,
        N_cell_x,
        r_cell_x,
        N_cell_y,
        r_cell_y,
        N_cell_z,
        r_cell_z,
    )

    # Calculate the initial accelerations for the particles assigned to this rank.
    for i in range(start_elems[rank], fin_elems[rank]):
        D = np.zeros(N_p)
        D[i] = FINITE_DIFFERENCE_STEP

        neighbor_list = form_Neib(
            X[i],
            Y[i],
            Z[i],
            cell_list,
            Lx,
            Ly,
            Lz,
            r_cell_x,
            r_cell_y,
            r_cell_z,
            N_cell_x,
            N_cell_y,
            N_cell_z,
        )

        Ui_dx = Ui(X + D, Y, Z, i, neighbor_list, R_off)
        Ui_dy = Ui(X, Y + D, Z, i, neighbor_list, R_off)
        Ui_dz = Ui(X, Y, Z + D, i, neighbor_list, R_off)
        Ui_0 = Ui(X, Y, Z, i, neighbor_list, R_off)

        Ax_old[i] = -(Ui_dx - Ui_0) / (
            UNIT_CONVERSION * ARGON_MASS * FINITE_DIFFERENCE_STEP
        )
        Ay_old[i] = -(Ui_dy - Ui_0) / (
            UNIT_CONVERSION * ARGON_MASS * FINITE_DIFFERENCE_STEP
        )
        Az_old[i] = -(Ui_dz - Ui_0) / (
            UNIT_CONVERSION * ARGON_MASS * FINITE_DIFFERENCE_STEP
        )

    if rank != 0:
        comm.send(Ax_old[start_elems[rank]:fin_elems[rank]], dest=0, tag=2)
        comm.send(Ay_old[start_elems[rank]:fin_elems[rank]], dest=0, tag=3)
        comm.send(Az_old[start_elems[rank]:fin_elems[rank]], dest=0, tag=4)
    else:
        for p in range(1, numprocs):
            Ax_old[start_elems[p]:fin_elems[p]] = comm.recv(source=p, tag=2)
            Ay_old[start_elems[p]:fin_elems[p]] = comm.recv(source=p, tag=3)
            Az_old[start_elems[p]:fin_elems[p]] = comm.recv(source=p, tag=4)

    Ax_old = comm.bcast(Ax_old, root=0)
    Ay_old = comm.bcast(Ay_old, root=0)
    Az_old = comm.bcast(Az_old, root=0)

    Ax = np.zeros(N_p)
    Ay = np.zeros(N_p)
    Az = np.zeros(N_p)

    for _ in range(N_steps):
        # Reflect velocities at the rigid walls.
        for i in range(start_elems[rank], fin_elems[rank]):
            if X[i] > Lx:
                Vx[i] = -abs(Vx[i])
            if X[i] < 0:
                Vx[i] = abs(Vx[i])

            if Y[i] > Ly:
                Vy[i] = -abs(Vy[i])
            if Y[i] < 0:
                Vy[i] = abs(Vy[i])

            if Z[i] > Lz:
                Vz[i] = -abs(Vz[i])
            if Z[i] < 0:
                Vz[i] = abs(Vz[i])

        if rank != 0:
            comm.send(Vx[start_elems[rank]:fin_elems[rank]], dest=0, tag=5)
            comm.send(Vy[start_elems[rank]:fin_elems[rank]], dest=0, tag=6)
            comm.send(Vz[start_elems[rank]:fin_elems[rank]], dest=0, tag=7)
        else:
            for p in range(1, numprocs):
                Vx[start_elems[p]:fin_elems[p]] = comm.recv(source=p, tag=5)
                Vy[start_elems[p]:fin_elems[p]] = comm.recv(source=p, tag=6)
                Vz[start_elems[p]:fin_elems[p]] = comm.recv(source=p, tag=7)

        Vx = comm.bcast(Vx, root=0)
        Vy = comm.bcast(Vy, root=0)
        Vz = comm.bcast(Vz, root=0)

        # Update coordinates.
        X += Vx * dt + 0.5 * Ax_old * dt**2
        Y += Vy * dt + 0.5 * Ay_old * dt**2
        Z += Vz * dt + 0.5 * Az_old * dt**2

        # Rebuild the cell list.
        cell_list = form_cells(
            X,
            Y,
            Z,
            Lx,
            Ly,
            Lz,
            N_cell_x,
            r_cell_x,
            N_cell_y,
            r_cell_y,
            N_cell_z,
            r_cell_z,
        )

        # Calculate new accelerations for the particles assigned to this rank.
        for i in range(start_elems[rank], fin_elems[rank]):
            D = np.zeros(N_p)
            D[i] = FINITE_DIFFERENCE_STEP

            neighbor_list = form_Neib(
                X[i],
                Y[i],
                Z[i],
                cell_list,
                Lx,
                Ly,
                Lz,
                r_cell_x,
                r_cell_y,
                r_cell_z,
                N_cell_x,
                N_cell_y,
                N_cell_z,
            )

            U_dx = Ui(X + D, Y, Z, i, neighbor_list, R_off)
            U_dy = Ui(X, Y + D, Z, i, neighbor_list, R_off)
            U_dz = Ui(X, Y, Z + D, i, neighbor_list, R_off)
            U_0 = Ui(X, Y, Z, i, neighbor_list, R_off)

            Ax[i] = -(U_dx - U_0) / (
                UNIT_CONVERSION * ARGON_MASS * FINITE_DIFFERENCE_STEP
            )
            Ay[i] = -(U_dy - U_0) / (
                UNIT_CONVERSION * ARGON_MASS * FINITE_DIFFERENCE_STEP
            )
            Az[i] = -(U_dz - U_0) / (
                UNIT_CONVERSION * ARGON_MASS * FINITE_DIFFERENCE_STEP
            )

        if rank != 0:
            comm.send(Ax[start_elems[rank]:fin_elems[rank]], dest=0, tag=8)
            comm.send(Ay[start_elems[rank]:fin_elems[rank]], dest=0, tag=9)
            comm.send(Az[start_elems[rank]:fin_elems[rank]], dest=0, tag=10)
        else:
            for p in range(1, numprocs):
                Ax[start_elems[p]:fin_elems[p]] = comm.recv(source=p, tag=8)
                Ay[start_elems[p]:fin_elems[p]] = comm.recv(source=p, tag=9)
                Az[start_elems[p]:fin_elems[p]] = comm.recv(source=p, tag=10)

        Ax = comm.bcast(Ax, root=0)
        Ay = comm.bcast(Ay, root=0)
        Az = comm.bcast(Az, root=0)

        # Update velocities.
        Vx += 0.5 * (Ax + Ax_old) * dt
        Vy += 0.5 * (Ay + Ay_old) * dt
        Vz += 0.5 * (Az + Az_old) * dt

        Ax_old = Ax.copy()
        Ay_old = Ay.copy()
        Az_old = Az.copy()

    return X, Y, Z, Vx, Vy, Vz


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------

def main():
    """Initialize the simulation and run the MPI-parallel MD calculation."""
    global N_p, Lx, Ly, Lz
    global r_cell_x, r_cell_y, r_cell_z
    global N_cell_x, N_cell_y, N_cell_z, R_off

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()

    print(f"Process {rank} out of {numprocs} started.", flush=True)

    # Simulation parameters.
    N_proposed = 10000
    Lx, Ly, Lz = 200, 200, 200
    V = 3.0

    # Initialization parameters.
    N_p = 0

    if rank == 0:
        X, Y, Z, Vx, Vy, Vz = initial_cond(
            N_proposed, Lx, Ly, Lz, R_MIN, V
        )
        N_p = len(X)

    N_p = comm.bcast(N_p, root=0)

    if rank != 0:
        X = np.zeros(N_p)
        Y = np.zeros(N_p)
        Z = np.zeros(N_p)
        Vx = np.zeros(N_p)
        Vy = np.zeros(N_p)
        Vz = np.zeros(N_p)

    X = comm.bcast(X, root=0)
    Y = comm.bcast(Y, root=0)
    Z = comm.bcast(Z, root=0)
    Vx = comm.bcast(Vx, root=0)
    Vy = comm.bcast(Vy, root=0)
    Vz = comm.bcast(Vz, root=0)

    N_p = len(X)

    # Construct the spatial cell grid.
    N_cell_x = int(Lx / R_OFF)
    r_cell_x = Lx / N_cell_x
    N_cell_y = int(Ly / R_OFF)
    r_cell_y = Ly / N_cell_y
    N_cell_z = int(Lz / R_OFF)
    r_cell_z = Lz / N_cell_z

    # Integration parameters.
    dt = 0.01
    N_steps = 20
    N_t = 2
    E_total = np.zeros(N_t)

    # Distribute particles between MPI processes.
    start_elems = np.zeros(numprocs, dtype=np.int64)
    fin_elems = np.zeros(numprocs, dtype=np.int64)
    elements_per_process = math.ceil(N_p / numprocs)

    for p in range(numprocs):
        start_elems[p] = (
            p * elements_per_process
            if p * elements_per_process < N_p
            else N_p
        )
        fin_elems[p] = (
            (p + 1) * elements_per_process
            if (p + 1) * elements_per_process < N_p
            else N_p
        )

    if rank == 0:
        start_time = time.time()

    for i in range(N_t):
        X, Y, Z, Vx, Vy, Vz = quick_Verlet_par(
            X,
            Y,
            Z,
            Vx,
            Vy,
            Vz,
            N_steps,
            dt,
            start_elems,
            fin_elems,
            rank,
            numprocs,
            comm,
        )

        if rank == 0:
            E_total[i] = E_tot(X, Y, Z, Vx, Vy, Vz)
            print(
                f"Iteration {i + 1} out of {N_t} done. "
                f"Energy = {E_total[i]}",
                flush=True,
            )

    if rank == 0:
        fin_time = time.time()
        print(f"Time: {fin_time - start_time}", flush=True)


if __name__ == "__main__":
    main()