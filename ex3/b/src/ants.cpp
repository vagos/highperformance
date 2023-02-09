#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <algorithm>

#include "utils.hpp"
#include "timer.hpp"

#define W N_cells
#define S (N_cells*N_cells)

#define STRING(x) #x

#ifndef LOG
#define LOG ""
#endif

// #define DEBUG

int steps = 100;

struct Colony
{
    const int N_cells, N_ants;
    const double p_inc, p_dec;

    int *A, *A_new;
    double *P, *P_new;
    int step = 0;

    Colony(int N_cells, int N_ants, double p_inc, double p_dec):
        N_cells(N_cells), N_ants(N_ants), p_inc(p_inc), p_dec(p_dec)
    {
        A   = (int*)std::calloc(S, sizeof(int));
        A_new = (int*)std::calloc(S, sizeof(int));

        P   = (double*)std::calloc(S, sizeof(double));
        P_new = (double*)std::calloc(S, sizeof(double));

        init();

        std::copy(A,A + S, A_new);
        std::copy(P,P + S, P_new);

#pragma acc enter data copyin(this)
#pragma acc enter data copyin(A[0:S], A_new[0:S], P[0:S], P_new[0:S], N_cells)
    }

    ~Colony()
    {
#pragma acc exit data delete(A[0:S], A_new[0:S], P[0:S], P_new[0:S])
        free(A);
        free(A_new);
        free(P);
        free(P_new);
    }

    void show(std::ostream& out_stream)
    {
#pragma acc update host(A[0:S], P[0:S])
        const std::string seperator(2 * N_cells, '=');

        out_stream << "STEP: " << step << '\n';

        out_stream << "ANTS in colony: " << "\n";
        
        out_stream << seperator << '\n';
        printArray(out_stream, A, N_cells, N_cells);
        out_stream << seperator << '\n';

        out_stream << "# ANTS: " << std::count(A, A + S, 1) << '\n';

        out_stream << "PHEROMONES in colony: " << "\n";

        out_stream << seperator << '\n';
        printArray(out_stream, P, N_cells, N_cells);
        out_stream << seperator << '\n';
    }

    void draw()
    {
        std::system("clear");
        printArray(std::cout, A, N_cells, N_cells);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    void update()
    {
        const int w = N_cells;
        const int h = N_cells;
    
        const int N_cells = this->N_cells;
        const double p_inc = this->p_inc;
        const double p_dec = this->p_dec;

#pragma acc parallel loop present(A_new[0:S])
        for (int i = 0; i < S; i++)
            A_new[i] = 0;

#pragma acc kernels present(this[0:1], A[0:S], A_new[0:S], P[0:S], P_new[0:S])
        {
        for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
                
                if (A[IDX(x, y)] == 1) 
                {
                    P_new[IDX(x, y)] = P[IDX(x, y)] * (1 + p_inc);
                }
                else // no ant
                {
                    P_new[IDX(x, y)] = P[IDX(x, y)] * (1 - p_dec);
                    continue; 
                }
                
                int best_idx = -1;
                double best_p = -1.0;

                best_cell(best_idx, best_p, x + 1, y);
                best_cell(best_idx, best_p, x - 1, y);
                best_cell(best_idx, best_p, x, y + 1);
                best_cell(best_idx, best_p, x, y - 1);

                assert(best_idx > 0);

                // ant moves to best_idx
                A_new[best_idx] = 1;
            }
        }

        } /* Kernels */

#pragma acc parallel loop present(this[0:1], A[0:S], A_new[0:S], P[0:S], P_new[0:S])
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {

                if (!(A_new[IDX(x, y)] == 0 && A[IDX(x, y)] == 1)) continue; // ant didn't move

                double current_pheromone = P[IDX(x, y)];

                P_new[IDX(x, y)] = current_pheromone / 2;

                if (x + 1 < N_cells) 
                       P_new[IDX(x + 1, y)] += current_pheromone / 8;
                if (x - 1 >= 0) 
                    P_new[IDX(x - 1, y)] += current_pheromone / 8;
                if (y + 1 < N_cells) 
                    P_new[IDX(x, y + 1)] += current_pheromone / 8;
                if (y - 1 >= 0) 
                    P_new[IDX(x, y - 1)] += current_pheromone / 8;

            }
        }
                
        std::swap(A, A_new);
        std::swap(P, P_new);
        step++;
    }

    private:
    void init()
    {
        static std::mt19937 generator;
        generator.seed(42);
        
        std::uniform_real_distribution<double> pheromone_placement(0, 1.0);

        std::vector<int> ant_placement(S);
        std::iota(ant_placement.begin(), ant_placement.end(), 0);
        std::shuffle(ant_placement.begin(), ant_placement.end(), generator);
        
        // init ants
        for (int a = 0; a < N_ants; a++)
        {
            A[ant_placement[a]] = 1;
        }

        // init pheromones
        for (int i = 0; i < N_cells * N_cells; i++)
        {
            P[i] = pheromone_placement(generator);
        }

    }

#pragma acc routine seq
    void best_cell(int& best_idx, double& best_p, int x, int y)
    {
        if (x < 0 || x >= N_cells || y < 0 || y >= N_cells)
            return;

        if (A_new[IDX(x, y)] == 1) // ant occupies cell
            return; 

        double current_p = P[IDX(x, y)];
        if (current_p > best_p)
        {
            best_p = current_p;
            best_idx = IDX(x, y);
        }
    }

};


int main (int argc, char *argv[])
{
    double p_inc = 0.1;
    double p_dec = 0.1;
    int N_cells = 1024;
    int N_ants = 100;


    if ((argc != 1) && (argc > 7)) {
        std::cout << "Usage: " << argv[0] << " -a <number-ants> -n <grid-size> -s <steps> [-h]" << std::endl;
        exit(EXIT_FAILURE);
    }

    for(int i = 1; i < argc; i++ ) {
        if( strcmp( argv[i], "-a" ) == 0 ) {
            N_ants = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-n" ) == 0 ) {
            N_cells = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-s" ) == 0 ) {
            steps = atoi(argv[i+1]);
            i++;
        }

        if( strcmp( argv[i], "-h" ) == 0 ) {
            std::cout << "Usage: " << argv[0] << " -a <number-ants> -n <grid-size> -s <steps> [-h]" << std::endl;
            exit(EXIT_SUCCESS);
        }
    }

    std::clog << "Starting with: "
        << S << " cells " << "| " 
        << N_ants << " ants" << " | "
        << steps << " steps" << std::endl;

    Colony* colony = new Colony(N_cells, N_ants, p_inc, p_dec); 

    std::ofstream log_file;
    log_file.open(std::string("simulation_") + std::string(LOG) + std::string(".log"), std::ios::out | std::ios::trunc);

    timer tm;

    tm.start();

    for (int s = 0; s < steps; s++)
    {
#ifdef DEBUG
        colony->show(log_file);
        // colony->draw();
#endif
        colony->update();
    }

    tm.stop();

    std::clog << "Time: " << tm.get_timing() << std::endl;

    colony->show(log_file);

    free(colony);
    return 0;
}
