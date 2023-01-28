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

// #define DEBUG

int steps = 100;

struct Colony
{
    int N_cells, N_ants;
    double p_inc, p_dec;

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
    }

    ~Colony()
    {
        free(A);
        free(A_new);
        free(P);
        free(P_new);
    }

    void show(std::ostream& out_stream)
    {
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
        int w = N_cells;
        int h = N_cells;
    
#pragma acc data create(A_new) create(P_new)
for (int s = 0; s < steps; s++) 
{

        for (int i = 0; i < S; i++)
            A_new[i] = 0;

#pragma acc data
#pragma acc kernels
        {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (A[IDX(x, y)] == 1) 
                {
                    P_new[IDX(x, y)] = P[IDX(x, y)] * (1 + p_inc);
                }
                else // no ant
                {
                    P_new[IDX(x, y)] = P[IDX(x, y)] * (1 - p_inc);
                    continue; 
                }
                
                int best_idx = -1;
                double best_p = -1.0;

                best_cell(best_idx, best_p, x + 1, y);
                best_cell(best_idx, best_p, x - 1, y);
                best_cell(best_idx, best_p, x, y + 1);
                best_cell(best_idx, best_p, x, y - 1);

                // std::clog << "Moving to: " << "(" << x << "," << y << ")" << "\n";

                if (best_idx < 0)
                {
                    A_new[IDX(x, y)] = 1;
                    continue; // ant sits still
                }
                
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


                // A_n[IDX(x, y)] = 0;
                A_new[best_idx] = 1;
            }
        }
        }

        std::swap(P, P_new);
        step++;
}
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


    inline void best_cell(int& best_idx, double& best_p, int x, int y)
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
    int N_cells = 2056;
    int N_ants = 100;


    if ((argc != 1) && (argc > 4)) {
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
    log_file.open("simulation.log", std::ios::out | std::ios::trunc);

    timer tm;

    tm.start();

//    for (int s = 0; s < steps; s++)
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
