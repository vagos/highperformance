#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

void print_array(int w, int h, double* array) 
{
    for (int j = 0; j < h; j++) {
    for (int i = 0; i < w; i++) {
        printf("%d ", (int)array[j*w + i]);
    }
    printf("\n");
    }
}

#define DEBUG 0

typedef struct Diagnostics_s
{
    double time;
    double heat;
} Diagnostics;

typedef struct Diffusion2D_s
{
    double D_, L_, T_;
    int N_, Ntot_, real_N_, grid_size;
    double dr_, dt_, fac_;
    int rank_, procs_;
    int local_N_;
    int square_N;
    double *rho_, *rho_tmp_;

    double *rho_left, *rho_right, *rho_up, *rho_down;
    double *rho_left_recv, *rho_right_recv, *rho_up_recv, *rho_down_recv;

    Diagnostics *diag_;

} Diffusion2D;

void initialize_density(Diffusion2D *D2D)
{
    int real_N_ = D2D->real_N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;
    double dr_ = D2D->dr_;
    double L_ = D2D->L_;
    int rank_ = D2D->rank_;
    int grid_size = D2D->grid_size;
    int gi, gj;

    /// Initialize rho(x, y, t=0).
    double bound = 0.25 * L_;

    for (int i = 1; i <= local_N_; ++i) {
        gi = (rank_ / grid_size) * local_N_ + i; // convert local index to global index
        for (int j = 1; j <= local_N_; ++j) {
            gj = (rank_ % grid_size) * local_N_ + j;
            if (fabs((gi - 1)*dr_ - 0.5*L_) < bound && fabs((gj - 1)*dr_ - 0.5*L_) < bound) {
                rho_[i*real_N_ + j] = 1;
            } else {
                rho_[i*real_N_ + j] = 0;
            }
        }
    }
}

void init(Diffusion2D *D2D,
                const double D,
                const double L,
                const int N,
                const int T,
                const double dt,
                const int rank,
                const int procs)
{
    D2D->grid_size = (int)sqrt(procs);

    int grid_size = D2D->grid_size;

    D2D->D_ = D;
    D2D->L_ = L;
    D2D->N_ = N;
    D2D->T_ = T;
    D2D->dt_ = dt;
    D2D->rank_ = rank;
    D2D->procs_ = procs;
    D2D->square_N = N / grid_size;

    // Real space grid spacing.
    D2D->dr_ = D2D->L_ / (D2D->N_ - 1);

    // Stencil factor.
    D2D->fac_ = D2D->dt_ * D2D->D_ / (D2D->dr_ * D2D->dr_);

    // Number of rows per process.
    D2D->local_N_ = D2D->square_N;

    // Small correction for the last process. (Not needed)
    // if (D2D->rank_ == D2D->procs_ - 1)
    //     D2D->local_N_ += D2D->N_ % D2D->procs_;

    // Actual dimension of a row (+2 for the ghost cells).
    D2D->real_N_ = D2D->square_N + 2;

    // Total number of cells.
    D2D->Ntot_ = (D2D->real_N_) * (D2D->real_N_);

    D2D->rho_ = (double *)calloc(D2D->Ntot_, sizeof(double));
    D2D->rho_tmp_ = (double *)calloc(D2D->Ntot_, sizeof(double));
    D2D->diag_ = (Diagnostics *)calloc(D2D->T_, sizeof(Diagnostics));

    D2D->rho_left = (double*)calloc(D2D->square_N, sizeof(double));
    D2D->rho_right = (double*)calloc(D2D->square_N, sizeof(double));
    D2D->rho_left_recv = (double*)calloc(D2D->square_N, sizeof(double));
    D2D->rho_right_recv = (double*)calloc(D2D->square_N, sizeof(double));

    // Check that the timestep satisfies the restriction for stability.
    if (D2D->rank_ == 0)
        printf("timestep from stability condition is %e\n", D2D->dr_ * D2D->dr_ / (4.0 * D2D->D_));

    initialize_density(D2D);

    // print_array(D2D->real_N_, D2D->real_N_, D2D->rho_);
}

void advance(Diffusion2D *D2D)
{
    int real_N_ = D2D->real_N_;
    double *rho_ = D2D->rho_;
    double *rho_tmp_ = D2D->rho_tmp_;
    double fac_ = D2D->fac_;
    int rank_ = D2D->rank_;
    int square_N = D2D->square_N;
    int grid_size = D2D->grid_size;

    MPI_Status status[4];

    int up_rank = rank_ - grid_size;
    int down_rank = rank_ + grid_size;
    int left_rank = rank_ - 1;
    int right_rank = rank_ + 1;

    double *rho_left = D2D->rho_left;
    double *rho_right = D2D->rho_right;
    double *rho_up = D2D->rho_up;
    double *rho_down = D2D->rho_down;
    
    double *rho_left_recv = D2D->rho_left_recv;
    double *rho_right_recv = D2D->rho_right_recv;
    double *rho_up_recv = D2D->rho_up_recv;
    double *rho_down_recv = D2D->rho_down_recv;

    // Fill the send buffers 

    rho_up = &rho_[1*real_N_+1];
    rho_down = &rho_[square_N*real_N_+1];

    rho_up_recv = &rho_[0*real_N_+1];
    rho_down_recv = &rho_[(square_N + 1)*real_N_+1];

    // Send and Receive

    if (rank_ / grid_size != grid_size - 1) { // square below exists
        MPI_Send(rho_down, square_N, MPI_DOUBLE, down_rank, 100, MPI_COMM_WORLD);
        MPI_Recv(rho_down_recv, square_N, MPI_DOUBLE, down_rank, 100, MPI_COMM_WORLD, &status[1]);
    }

    if (rank_ / grid_size != 0) { // square above exists
        MPI_Recv(rho_up_recv, square_N, MPI_DOUBLE, up_rank, 100, MPI_COMM_WORLD, &status[0]);
        MPI_Send(rho_up, square_N, MPI_DOUBLE, up_rank, 100, MPI_COMM_WORLD);
    }

    if (rank_ % grid_size != grid_size - 1) { // square right exists

        for (int i = 1; i <= square_N; ++i) {
            rho_right[i - 1] = rho_[i*real_N_ + square_N];
        }

        MPI_Send(rho_right, square_N, MPI_DOUBLE, right_rank, 200, MPI_COMM_WORLD);
        MPI_Recv(rho_right_recv, square_N, MPI_DOUBLE, right_rank, 200, MPI_COMM_WORLD, &status[2]);

        for (int i = 1; i <= square_N; ++i) {
            rho_[i*real_N_ + square_N + 1] = rho_right_recv[i - 1];
        }
    }

    if (rank_ % grid_size != 0) { // square left exists

        for (int i = 1; i <= square_N; ++i) {
            rho_left[i - 1] = rho_[i*real_N_ + 1];
        }

        MPI_Recv(rho_left_recv, square_N, MPI_DOUBLE, left_rank, 200, MPI_COMM_WORLD, &status[3]);
        MPI_Send(rho_left, square_N, MPI_DOUBLE, left_rank, 200, MPI_COMM_WORLD);

        for (int i = 1; i <= square_N; ++i) {
            rho_[i*real_N_ + 0] = rho_left_recv[i - 1];
        }
    }


    // Exchange ALL necessary ghost cells with neighboring ranks.

    // Central differences in space, forward Euler in time with Dirichlet
    // boundaries.

    // Adding thread parallelization for each process makes the program a lot slower for the given grid sizes (1024, 2048, 4096).
    // #pragma omp parallel for
    for (int i = 1; i <= square_N; ++i) { // y 
        for (int j = 1; j <= square_N; ++j) { // x
            rho_tmp_[i*real_N_ + j] = rho_[i*real_N_ + j] +
                                     fac_
                                     *
                                     (
                                     + rho_[i*real_N_ + (j+1)]
                                     + rho_[i*real_N_ + (j-1)]
                                     + rho_[(i+1)*real_N_ + j]
                                     + rho_[(i-1)*real_N_ + j]
                                     - 4.*rho_[i*real_N_ + j]
                                     );
        }
    }

    // Swap rho_ with rho_tmp_. This is much more efficient,
    // because it does not copy element by element, just replaces storage
    // pointers.
    double *tmp_ = D2D->rho_tmp_;
    D2D->rho_tmp_ = D2D->rho_;
    D2D->rho_ = tmp_;
}

void compute_diagnostics(Diffusion2D *D2D, const int step, const double t)
{
    int real_N_ = D2D->real_N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;
    double dr_ = D2D->dr_;
    int rank_ = D2D->rank_;

    double heat = 0.0;
    for(int i = 1; i <= local_N_; ++i)
        for(int j = 1; j <= local_N_; ++j)
            heat += rho_[i*real_N_ + j] * dr_ * dr_;

    // TODO:MPI, reduce heat (sum)
    MPI_Reduce(rank_ == 0? MPI_IN_PLACE: &heat, &heat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 


    if (rank_ == 0) {
#if DEBUG
        printf("t = %lf heat = %lf\n", t, heat);
#endif
        D2D->diag_[step].time = t;
        D2D->diag_[step].heat = heat;
    }
}

void write_diagnostics(Diffusion2D *D2D, const char *filename)
{

    FILE *out_file = fopen(filename, "w");
    for (int i = 0; i < D2D->T_; i++)
        fprintf(out_file, "%lf\t%lf\n", D2D->diag_[i].time, D2D->diag_[i].heat);
    fclose(out_file);
}


int main(int argc, char* argv[])
{
    if (argc < 6) {
        printf("Usage: %s D L T N dt\n", argv[0]);
        return 1;
    }

    int rank, procs;
    //TODO:MPI Initialize MPI, number of ranks (rank) and number of processes (nprocs) involved in the communicator
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    const double D = atof(argv[1]);
    const double L = atoi(argv[2]);
    const int N = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const double dt = atof(argv[5]);

    Diffusion2D system;

    init(&system, D, L, N, T, dt, rank, procs);

    double t0 = MPI_Wtime();
    for (int step = 0; step < T; ++step) {
#ifndef _PERF_
        compute_diagnostics(&system, step, dt * step);
#endif
        advance(&system);
    }
    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("Timing: %d %lf\n", N, t1-t0);

#ifndef _PERF_
    if (rank == 0) {
        char diagnostics_filename[256];
        sprintf(diagnostics_filename, "diagnostics_mpi_%d.dat", procs);
        write_diagnostics(&system, diagnostics_filename);
    }
#endif

    MPI_Finalize();
    return 0;
}
