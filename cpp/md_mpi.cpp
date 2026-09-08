/**
 * Molecular Dynamics simulation using MPI.
 *
 * This program implements a short-range Lennard-Jones interaction model
 * and advances particle positions and velocities with a velocity-Verlet
 * integration scheme. Particle accelerations are evaluated numerically
 * using finite differences. MPI is used to distribute particles among
 * processes during the acceleration calculation.
 *
 * The code is intended for reproducible performance experiments comparing
 * parallel implementations of the same molecular-dynamics algorithm.
 *
 * NOTE:
 * The numerical algorithm and model parameters are intentionally preserved
 * from the original research code. In particular, the energy calculation
 * currently uses its original local cutoff value.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include "mpi.h"

using std::vector;
using std::cout;
using std::endl;
using std::sqrt;
using std::pow;
using std::floor;
using std::ceil;
using std::min;
using std::abs;

const double PI = acos(-1);

int N_p;
const double m1 = 39.938;
const double r_lim = 3.822; // Minimum allowed initial inter-particle distance
const double r_min = 3.822;
const double R_on = 2.0 * r_min;
const double R_off = 2.5 * r_min;
const double C_units = 9648.533977;
const double Lx = 200, Ly = 200, Lz = 200; // Simulation box dimensions
const double N_cell_x = floor(Lx / R_off), N_cell_y = floor(Ly / R_off), N_cell_z = floor(Lz / R_off);
const double r_cell_x = Lx / N_cell_x, r_cell_y = Ly / N_cell_y, r_cell_z = Lz / N_cell_z;


void initial_cond(int& N, double Lx, double Ly, double Lz, double r_lim, double V, vector<double>& X, vector<double>& Y, vector<double>& Z, vector<double>& Vx, vector<double>& Vy, vector<double>& Vz) {
    srand((unsigned int)time(0));
    X[0] = Lx * (double)rand() / (RAND_MAX);
    Y[0] = Ly * (double)rand() / (RAND_MAX);
    Z[0] = Lz * (double)rand() / (RAND_MAX);

    int failed_to_place_N_particles = 0;

    int i = 0;
    for (i = 1; i < N; i++) {
        X[i] = Lx * (double)rand() / (RAND_MAX);
        Y[i] = Ly * (double)rand() / (RAND_MAX);
        Z[i] = Lz * (double)rand() / (RAND_MAX);

        int k = 0;
        int err = 1;

        while ((err == 1) and (k <= 10)) {

            err = 0;

            for (int j = 0; j < i; j++) {
                double R = sqrt((X[i] - X[j]) * (X[i] - X[j]) + (Y[i] - Y[j]) * (Y[i] - Y[j]) + (Z[i] - Z[j]) * (Z[i] - Z[j]));

                if (R < r_lim) {
                    err = 1;
                    break;
                }
            }

            if (err == 1) {
                X[i] = Lx * (double)rand() / (RAND_MAX);
                Y[i] = Ly * (double)rand() / (RAND_MAX);
                Z[i] = Lz * (double)rand() / (RAND_MAX);
                k++;
            }
        }

        if (k >= 10) {
            cout << "Failed to place " << N << " particles. Only " << i << " particles have been placed." << endl;
            failed_to_place_N_particles = 1;
            break;
        }
    }

    if (failed_to_place_N_particles == 1) {

        X = vector<double>(X.begin(), X.begin() + i);
        Y = vector<double>(Y.begin(), Y.begin() + i);
        Z = vector<double>(Z.begin(), Z.begin() + i);

        Vx = vector<double>(Vx.begin(), Vx.begin() + i);
        Vy = vector<double>(Vy.begin(), Vy.begin() + i);
        Vz = vector<double>(Vz.begin(), Vz.begin() + i);
    }

    N = (int)X.size();


    for (i = 0; i < N; i++) {

        double phi = 2 * PI * (double)rand() / (RAND_MAX);
        double theta = PI * (double)rand() / (RAND_MAX);

        Vx[i] = V * cos(phi) * sin(theta);
        Vy[i] = V * sin(phi) * sin(theta);
        Vz[i] = V * cos(theta);
    }
}



/** Compute the smoothed and truncated Lennard-Jones potential. */
double U_lj_cut(double r) {

    double w, epsilon = .01;

    if (r < R_on)
        w = 1;
    else if (r > R_off)
        w = 0;
    else
        w = pow((R_off * R_off - r * r), 2) * (R_off * R_off - 3 * (R_on * R_on) + 2 * r * r) / pow((R_off * R_off - R_on * R_on), 3);

    return w * epsilon * (pow(r_min / r, 12) - 2 * pow(r_min / r, 6));
}



/** Compute the potential energy contribution of one particle. */
double Ui(const vector<double>& X, const vector<double>& Y, const vector<double>& Z, int i, const vector<int>& Neib_list, double R_off) {

    size_t N_neib = Neib_list.size();
    double s = 0.0;

    for (int j = 0; j < N_neib; ++j) {
        int k = Neib_list[j];

        double dx = X[i] - X[k];
        double dy = Y[i] - Y[k];
        double dz = Z[i] - Z[k];
        double r_ij = sqrt(dx * dx + dy * dy + dz * dz);

        if (r_ij < R_off && i != k)
            s += U_lj_cut(r_ij);

    }

    return s;
}



/** Compute the total potential energy of the system. */
double U_tot(const vector<double>& X, const vector<double>& Y, const vector<double>& Z) {

    double R_cut = 2.5;
    double s = 0.0;
    size_t N = X.size();

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < i; ++j) {
            double R = sqrt(pow(X[i] - X[j], 2) + pow(Y[i] - Y[j], 2) + pow(Z[i] - Z[j], 2));
            if (R < R_cut) {
                s += U_lj_cut(R);
            }
        }
    }

    return s;
}



/** Compute the total kinetic plus potential energy. */
double E_tot(const vector<double>& X, const vector<double>& Y, const vector<double>& Z,
    const vector<double>& Vx, const vector<double>& Vy, const vector<double>& Vz) {

    double m1 = 39.938;
    double C_units = 9648.533977;
    double R_cut = 2.5;

    size_t N = X.size();


    double s1 = 0.0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < i; ++j) {
            double R = sqrt(pow(X[i] - X[j], 2) + pow(Y[i] - Y[j], 2) + pow(Z[i] - Z[j], 2));
            if (R < R_cut) {
                s1 += U_lj_cut(R);
            }
        }
    }


    double s2 = 0.0;
    for (size_t i = 0; i < N; ++i) {
        s2 += C_units * m1 * (pow(Vx[i], 2) + pow(Vy[i], 2) + pow(Vz[i], 2)) / 2;
    }


    return s1 + s2;
}



/** Build the spatial cell list used for short-range neighbor search. */
void form_cells(const vector<double>& X, const vector<double>& Y, const vector<double>& Z,
    double Lx, double Ly, double Lz,
    int N_cell_x, double r_cell_x, int N_cell_y, double r_cell_y, int N_cell_z, double r_cell_z, vector<vector<vector<vector<int>>>>& Cell_list) {

    // Assign particles to cells
    for (int i = 0; i < N_p; ++i) {
        int nx, ny, nz;

        if (X[i] > 0 && X[i] < Lx)
            nx = min(static_cast<int>(std::ceil(X[i] / r_cell_x)) - 1, N_cell_x - 1);
        else
            nx = (X[i] < 0) ? 0 : N_cell_x - 1;

        if (Y[i] > 0 && Y[i] < Ly)
            ny = min(static_cast<int>(std::ceil(Y[i] / r_cell_y)) - 1, N_cell_y - 1);
        else
            ny = (Y[i] < 0) ? 0 : N_cell_y - 1;

        if (Z[i] > 0 && Z[i] < Lz)
            nz = min(static_cast<int>(std::ceil(Z[i] / r_cell_z)) - 1, N_cell_z - 1);
        else
            nz = (Z[i] < 0) ? 0 : N_cell_z - 1;

        Cell_list[nx][ny][nz].push_back(i);
    }

}



/** Collect particles from the cell containing a particle and its neighbors. */
void form_Neib(double Xi, double Yi, double Zi, const vector<vector<vector<vector<int>>>>& Cell_list,
    double Lx, double Ly, double Lz,
    double r_cell_x, double r_cell_y, double r_cell_z,
    int N_cell_x, int N_cell_y, int N_cell_z, vector<int>& Neib_list) {

    // Determine the cell containing the particle
    int nx, ny, nz;

    if (Xi > 0 && Xi < Lx)
        nx = static_cast<int>(std::ceil(Xi / r_cell_x)) - 1;
    else
        nx = (Xi < 0) ? 0 : N_cell_x - 1;

    if (Yi > 0 && Yi < Ly)
        ny = static_cast<int>(std::ceil(Yi / r_cell_y)) - 1;
    else
        ny = (Yi < 0) ? 0 : N_cell_y - 1;

    if (Zi > 0 && Zi < Lz)
        nz = static_cast<int>(std::ceil(Zi / r_cell_z)) - 1;
    else
        nz = (Zi < 0) ? 0 : N_cell_z - 1;


    // Search neighboring cells
    for (int jx = -1; jx <= 1; ++jx) {
        for (int jy = -1; jy <= 1; ++jy) {
            for (int jz = -1; jz <= 1; ++jz) {

                // Check that the neighboring cell is inside the simulation box
                if ((nx + jx >= 0) && (nx + jx < N_cell_x) &&
                    (ny + jy >= 0) && (ny + jy < N_cell_y) &&
                    (nz + jz >= 0) && (nz + jz < N_cell_z)) {

                    // Add particles from the neighboring cell
                    const vector<int>& neighbor_particles = Cell_list[nx + jx][ny + jy][nz + jz];
                    Neib_list.insert(Neib_list.end(), neighbor_particles.begin(), neighbor_particles.end());
                }
            }
        }
    }
}



/** Run the MPI-parallel velocity-Verlet integration. */
void quick_Verlet_par(vector<double>& X, vector<double>& Y, vector<double>& Z,
    vector<double>& Vx, vector<double>& Vy, vector<double>& Vz, int N_steps, double dt,
    int start_el, int fin_el, int numtasks, int rank, vector<int>& slice_sizes) {


    double delta = 0.001;

    int slice_size = fin_el - start_el;

    // Acceleration vectors
    vector<double> Ax_old(slice_size, 0.0);
    vector<double> Ay_old(slice_size, 0.0);
    vector<double> Az_old(slice_size, 0.0);


    vector<vector<vector<vector<int>>>> Cell_list(N_cell_x, vector<vector<vector<int>>>(N_cell_y, vector<vector<int>>(N_cell_z)));
    form_cells(X, Y, Z, Lx, Ly, Lz, N_cell_x, r_cell_x, N_cell_y, r_cell_y, N_cell_z, r_cell_z, Cell_list);


    for (int i = start_el; i < fin_el; ++i) {

        vector<double> D(N_p, 0.0);
        D[i] = delta;

        vector<double> X_D(N_p, 0.0);
        vector<double> Y_D(N_p, 0.0);
        vector<double> Z_D(N_p, 0.0);
        for (int i = 0; i < N_p; ++i) {
            X_D[i] = X[i] + D[i];
            Y_D[i] = Y[i] + D[i];
            Z_D[i] = Z[i] + D[i];
        }

        vector<int> Neib_list;
        form_Neib(X[i], Y[i], Z[i], Cell_list, Lx, Ly, Lz, r_cell_x, r_cell_y, r_cell_z, N_cell_x, N_cell_y, N_cell_z, Neib_list);

        double Ui_dx = Ui(X_D, Y, Z, i, Neib_list, R_off);
        double Ui_dy = Ui(X, Y_D, Z, i, Neib_list, R_off);
        double Ui_dz = Ui(X, Y, Z_D, i, Neib_list, R_off);
        double Ui_0 = Ui(X, Y, Z, i, Neib_list, R_off);


        Ax_old[i - start_el] = -(Ui_dx - Ui_0) / (C_units * m1 * delta);
        Ay_old[i - start_el] = -(Ui_dy - Ui_0) / (C_units * m1 * delta);
        Az_old[i - start_el] = -(Ui_dz - Ui_0) / (C_units * m1 * delta);
    }

    MPI_Barrier(MPI_COMM_WORLD);


    if (rank != 0) {
        MPI_Send(Ax_old.data(), slice_sizes[rank], MPI_DOUBLE, 0, 42, MPI_COMM_WORLD);
        MPI_Send(Ay_old.data(), slice_sizes[rank], MPI_DOUBLE, 0, 43, MPI_COMM_WORLD);
        MPI_Send(Az_old.data(), slice_sizes[rank], MPI_DOUBLE, 0, 44, MPI_COMM_WORLD);
        Ax_old.resize(N_p);
        Ay_old.resize(N_p);
        Az_old.resize(N_p);
    }
    else {
        for (int i = 1; i < numtasks; i++) {
            vector<double> Ax_old_slice(slice_sizes[i]);
            vector<double> Ay_old_slice(slice_sizes[i]);
            vector<double> Az_old_slice(slice_sizes[i]);
            MPI_Recv(Ax_old_slice.data(), (int)Ax_old_slice.size(), MPI_DOUBLE, i, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(Ay_old_slice.data(), (int)Ay_old_slice.size(), MPI_DOUBLE, i, 43, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(Az_old_slice.data(), (int)Az_old_slice.size(), MPI_DOUBLE, i, 44, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Ax_old.insert(Ax_old.end(), Ax_old_slice.begin(), Ax_old_slice.end());
            Ay_old.insert(Ay_old.end(), Ay_old_slice.begin(), Ay_old_slice.end());
            Az_old.insert(Az_old.end(), Az_old_slice.begin(), Az_old_slice.end());
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(Ax_old.data(), (int)Ax_old.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Ay_old.data(), (int)Ay_old.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Az_old.data(), (int)Az_old.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);


    for (int j = 0; j < N_steps; ++j) {
        for (int i = 0; i < N_p; ++i) {
            if (X[i] > Lx)
                Vx[i] = -abs(Vx[i]);
            if (Y[i] < 0)
                Vy[i] = abs(Vy[i]);
            if (Y[i] > Ly)
                Vy[i] = -abs(Vy[i]);
            if (Y[i] < 0)
                Vy[i] = abs(Vy[i]);
            if (Z[i] > Lz)
                Vz[i] = -abs(Vz[i]);
            if (Z[i] < 0)
                Vz[i] = abs(Vz[i]);
        }

        for (int j = 0; j < N_p; ++j) {
            X[j] += Vx[j] * dt + 0.5 * Ax_old[j] * dt * dt;
            Y[j] += Vy[j] * dt + 0.5 * Ay_old[j] * dt * dt;
            Z[j] += Vz[j] * dt + 0.5 * Az_old[j] * dt * dt;
        }


        vector<double> Ax(slice_size, 0.0);
        vector<double> Ay(slice_size, 0.0);
        vector<double> Az(slice_size, 0.0);


        vector<vector<vector<vector<int>>>> Cell_list(N_cell_x, vector<vector<vector<int>>>(N_cell_y, vector<vector<int>>(N_cell_z)));
        form_cells(X, Y, Z, Lx, Ly, Lz, N_cell_x, r_cell_x, N_cell_y, r_cell_y, N_cell_z, r_cell_z, Cell_list);


        for (int i = start_el; i < fin_el; ++i) {
            vector<double> D(N_p, 0.0);
            D[i] = delta;

            vector<double> X_D(N_p, 0.0);
            vector<double> Y_D(N_p, 0.0);
            vector<double> Z_D(N_p, 0.0);
            for (int i = 0; i < N_p; ++i) {
                X_D[i] = X[i] + D[i];
                Y_D[i] = Y[i] + D[i];
                Z_D[i] = Z[i] + D[i];
            }


            vector<int> Neib_list;
            form_Neib(X[i], Y[i], Z[i], Cell_list, Lx, Ly, Lz, r_cell_x, r_cell_y, r_cell_z, N_cell_x, N_cell_y, N_cell_z, Neib_list);

            double U_dx = Ui(X_D, Y, Z, i, Neib_list, R_off);
            double U_dy = Ui(X, Y_D, Z, i, Neib_list, R_off);
            double U_dz = Ui(X, Y, Z_D, i, Neib_list, R_off);
            double U_0 = Ui(X, Y, Z, i, Neib_list, R_off);

            Ax[i - start_el] = -(U_dx - U_0) / (C_units * m1 * delta);
            Ay[i - start_el] = -(U_dy - U_0) / (C_units * m1 * delta);
            Az[i - start_el] = -(U_dz - U_0) / (C_units * m1 * delta);
        }


        MPI_Barrier(MPI_COMM_WORLD);


        if (rank != 0) {
            MPI_Send(Ax.data(), slice_sizes[rank], MPI_DOUBLE, 0, 45, MPI_COMM_WORLD);
            MPI_Send(Ay.data(), slice_sizes[rank], MPI_DOUBLE, 0, 46, MPI_COMM_WORLD);
            MPI_Send(Az.data(), slice_sizes[rank], MPI_DOUBLE, 0, 47, MPI_COMM_WORLD);
            Ax.resize(N_p);
            Ay.resize(N_p);
            Az.resize(N_p);
        }
        else {
            for (int i = 1; i < numtasks; i++) {
                vector<double> Ax_slice(slice_sizes[i]);
                vector<double> Ay_slice(slice_sizes[i]);
                vector<double> Az_slice(slice_sizes[i]);
                MPI_Recv(Ax_slice.data(), (int)Ax_slice.size(), MPI_DOUBLE, i, 45, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(Ay_slice.data(), (int)Ay_slice.size(), MPI_DOUBLE, i, 46, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(Az_slice.data(), (int)Az_slice.size(), MPI_DOUBLE, i, 47, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                Ax.insert(Ax.end(), Ax_slice.begin(), Ax_slice.end());
                Ay.insert(Ay.end(), Ay_slice.begin(), Ay_slice.end());
                Az.insert(Az.end(), Az_slice.begin(), Az_slice.end());
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Bcast(Ax.data(), (int)Ax.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(Ay.data(), (int)Ay.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(Az.data(), (int)Az.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);


        for (int j = 0; j < N_p; ++j) {
            Vx[j] += 0.5 * (Ax[j] + Ax_old[j] * dt);
            Vy[j] += 0.5 * (Ay[j] + Ay_old[j] * dt);
            Vz[j] += 0.5 * (Az[j] + Az_old[j] * dt);

            Ax_old[j] = Ax[j];
            Ay_old[j] = Ay[j];
            Az_old[j] = Az[j];
        }
    }
}



/** Program entry point. */
int main(int* argc, char** argv) {

    int numtasks, rank;

    MPI_Init(argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    cout << "Hello from process " << rank << " out of " << numtasks << endl;


    int N_proposed = 10000;
    double V = 3;
    double dt = 0.01;
    int N_steps = 20;
    int N_t = 2;
    vector<double> E_total(N_t, 0.);


    vector<double> X(N_proposed, 0.); vector<double> Y(N_proposed, 0.); vector<double> Z(N_proposed, 0.);
    vector<double> Vx(N_proposed, 0.); vector<double> Vy(N_proposed, 0.); vector<double> Vz(N_proposed, 0.);

    if (rank == 0) {
        initial_cond(N_proposed, Lx, Ly, Lz, r_lim, V, X, Y, Z, Vx, Vy, Vz);
        N_p = X.size();
    }
    MPI_Bcast(&N_p, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(X.data(), N_p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Y.data(), N_p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Z.data(), N_p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Vx.data(), N_p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Vy.data(), N_p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Vz.data(), N_p, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    vector<int> slice_sizes(numtasks, N_p / numtasks);
    for (int i = 0; i < N_p % numtasks; ++i)
        slice_sizes[i]++;


    int s = 0, start_el, fin_el;
    for (int i = 0; i < slice_sizes.size(); ++i) {
        if (rank == i) {
            start_el = s;
            fin_el = s + slice_sizes[i];
        }
        s += slice_sizes[i];
    }


    chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (int i = 0; i < N_t; i++) {

        quick_Verlet_par(X, Y, Z, Vx, Vy, Vz, N_steps, dt, start_el, fin_el, numtasks, rank, slice_sizes);
        E_total[i] = E_tot(X, Y, Z, Vx, Vy, Vz);

        if (rank == 0)
            cout << "Iteration: " << i + 1 << " out of " << N_t << " done. Energy = " << E_total[i] << endl;
    }


    chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (rank == 0)
        cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;


    MPI_Finalize();
}
