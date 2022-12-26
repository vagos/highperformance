#include "utils.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ostream>
#include <random>
#include <thread>

#define W N_cells
#define S (N_cells*N_cells)

struct Colony
{
    int N_cells, N_ants;
    double p_inc, p_dec;

    int *A, *A_n;
    double *P, *P_n;
    int step = 0;

    Colony(int N_cells, int N_ants, double p_inc, double p_dec):
        N_cells(N_cells), N_ants(N_ants), p_inc(p_inc), p_dec(p_dec)
    {
        A   = (int*)std::calloc(S, sizeof(int));
        A_n = (int*)std::calloc(S, sizeof(int));

        P   = (double*)std::calloc(S, sizeof(double));
        P_n = (double*)std::calloc(S, sizeof(double));

        init();

        std::copy(A,A + S, A_n);
        std::copy(P,P + S, P_n);
    }

    ~Colony()
    {
        free(A);
        free(A_n);
        free(P);
        free(P_n);
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
        // std::system("clear");
        printArray(std::cout, A, N_cells, N_cells);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    void update()
    {
        int w = N_cells;
        int h = N_cells;

        std::memset(A_n, 0, S*sizeof(int));

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (A[IDX(x, y)] == 1) 
                {
                    P_n[IDX(x, y)] = P[IDX(x, y)] * (1 + p_inc);
                }
                else // no ant
                {
                    P_n[IDX(x, y)] = P[IDX(x, y)] * (1 - p_inc);
                    continue; 
                }
                
                int best_idx = -1;
                double best_p = -1.0;

                best_cell(best_idx, best_p, x + 1, y);
                best_cell(best_idx, best_p, x - 1, y);
                best_cell(best_idx, best_p, x, y + 1);
                best_cell(best_idx, best_p, x, y - 1);

                std::clog << "Moving to: " << "(" << x << "," << y << ")" << "\n";

                // assert(best_idx > 0);

                if (best_idx < 0)
                {
                    A[IDX(x, y)] = 1;
                    continue; // ant sits still
                }
                
                double current_p = P[IDX(x, y)];
                
                P_n[IDX(x, y)] = current_p / 2;
                
                P_n[IDX(x + 1, y)] += current_p / 8;
                P_n[IDX(x - 1, y)] += current_p / 8;
                P_n[IDX(x, y + 1)] += current_p / 8;
                P_n[IDX(x, y - 1)] += current_p / 8;

                // A_n[IDX(x, y)] = 0;
                A_n[best_idx] = 1;
            }
        }

        std::swap(P, P_n);
        std::swap(A, A_n);

        step++;
    }

    private:
    void init()
    {
        // static thread_local std::mt19937 generator;
        static std::mt19937 generator;
        generator.seed(0);
        
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

        if (A_n[IDX(x, y)] == 1) // ant occupies cell
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
    int N_cells = 3;
    int N_ants = 2;
    double p_inc = 0.1;
    double p_dec = 0.1;
    int steps = 100;

    Colony* colony = new Colony(N_cells, N_ants, p_inc, p_dec); 

    std::ofstream log_file;
    log_file.open("simulation.log", std::ios::out | std::ios::trunc);

    for (int s = 0; s < steps; s++)
    {
        colony->show(log_file);
        // colony->draw();
        colony->update();
    }

    free(colony);

    return 0;
}
