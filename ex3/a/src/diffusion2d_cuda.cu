#include <iostream>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include "timer.hpp"

typedef std::size_t size_type;

static const int diffusion_block_x = 16;
static const int diffusion_block_y = 16;
cudaError_t err = cudaSuccess;

__global__ void diffusion_kernel(float * rho_out, float const * rho, float fac, int N)
{
	//compute rho_out i, j
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < N && j < N)
    {
        rho_out[i*N + j] =
                rho[i*N + j]
                +
                fac
                *
                (
                 (j == N-1 ? 0 : rho[i*N + (j+1)])
                 +
                 (j == 0 ? 0 : rho[i*N + (j-1)])
                 +
                 (i == N-1 ? 0 : rho[(i+1)*N + j])
                 +
                 (i == 0 ? 0 : rho[(i-1)*N + j])
                 -
                 4*rho[i*N + j]
                 );
        
    }
}

class Diffusion2D
{

public:

    Diffusion2D( const float D, const float rmax, const float rmin, const size_type N):
        D_(D),
        rmax_(rmax),
        rmin_(rmin),
        N_(N),
        N_tot(N*N),
        d_rho_(0),
        d_rho_tmp_(0),
        rho_(N_tot)
    {
        /// real space grid spacing
        dr_ = (rmax_ - rmin_) / (N_ - 1);

        /// dt < dx*dx / (4*D) for stability
        dt_ = dr_ * dr_ / (6 * D_);

        /// stencil factor
        fac_ = dt_ * D_ / (dr_ * dr_);

        // Allocate memory on Device

		//allocate d_rho_ and d_rho_tmp_ on the GPU and set them to zero
        err = cudaMalloc((void**)&d_rho_, N_tot*sizeof(float));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate vector rho (error code %s)!", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        cudaMalloc((void**)&d_rho_tmp_, N_tot*sizeof(float));

        err = cudaMemset(d_rho_, 0, N_tot*sizeof(float));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate vector rho (error code %s)!", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        cudaMemset(d_rho_tmp_, 0, N_tot*sizeof(float));


        InitializeSystem();
    }

    ~Diffusion2D()
    {
        cudaFree(d_rho_tmp_);
        cudaFree(d_rho_);
    }

    void PropagateDensity(int steps);

    float GetMoment();

    float GetTime() const {return time_;}

    void WriteDensity(const std::string file_name) const;

private:

    void InitializeSystem();

    const float D_, rmax_, rmin_;
    const size_type N_;
    size_type N_tot;

    float dr_, dt_, fac_;

    float time_;

    float *d_rho_, *d_rho_tmp_;
    mutable std::vector<float> rho_;
};

float Diffusion2D::GetMoment()
{
        //Get data (rho_) from the GPU device
        err = cudaMemcpy(&rho_[0], d_rho_, N_tot*sizeof(float), cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector rho from device to host (error code %s)!", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        float sum = 0;

        for(size_type i = 0; i < N_; ++i)
            for(size_type j = 0; j < N_; ++j) {
                float x = j*dr_ + rmin_;
                float y = i*dr_ + rmin_;
                sum += rho_[i*N_ + j] * (x*x + y*y);
            }

        return dr_*dr_*sum;
}

void Diffusion2D::WriteDensity(const std::string file_name) const
{
	//Get data (rho_) from the GPU device
    err = cudaMemcpy(&rho_[0], d_rho_, N_tot*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector rho from device to host (error code %s)!", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    std::ofstream out_file;
    out_file.open(file_name.c_str(), std::ios::out);
    if(out_file.good())
    {
        for(size_type i = 0; i < N_; ++i){
            for(size_type j = 0; j < N_; ++j)
                out_file << (i*dr_+rmin_) << '\t' << (j*dr_+rmin_) << '\t' << rho_[i*N_ + j] << "\n";

            out_file << "\n";
        }
    }

    out_file.close();
}

void Diffusion2D::PropagateDensity(int steps)
{
    using std::swap;
    /// Dirichlet boundaries; central differences in space, forward Euler
    /// in time

	//define grid_size and block_size
    int threads_x = diffusion_block_x;
    int threads_y = diffusion_block_y;
    dim3 threadsPerBlock(threads_x,threads_y);

    int blocks_x = (N_ + threads_x - 1) / threads_x;
    int blocks_y = (N_ + threads_y - 1) / threads_y;
    blocks_x = (blocks_x == 0)? 1 : blocks_x;
    blocks_y = (blocks_y == 0)? 1 : blocks_y;

    dim3 blocksPerGrid(blocks_x,blocks_y);

    for(int s = 0; s < steps; ++s)
    {
        diffusion_kernel<<< blocksPerGrid , threadsPerBlock >>>(d_rho_tmp_, d_rho_, fac_, N_);
        swap(d_rho_, d_rho_tmp_);
        time_ += dt_;
    }
}

void Diffusion2D::InitializeSystem()
{
    time_ = 0;

    /// initialize rho(x,y,t=0)
    float bound = 1./2;

    for(size_type i = 0; i < N_; ++i){
        for(size_type j = 0; j < N_; ++j){
            if(std::fabs(i*dr_+rmin_) < bound && std::fabs(j*dr_+rmin_) < bound){
                rho_[i*N_ + j] = 1;
            }
            else{
                rho_[i*N_ + j] = 0;
            }

        }
    }

	//Copy data to the GPU device
    err = cudaMemcpy(d_rho_, &rho_[0], N_tot * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector rho from host to device (error code %s)!", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " <log2(size)>" << std::endl;
        return 1;
    }

    const float D = 1;
    const float tmax = 0.01;
    const float rmax = 1;
    const float rmin = -1;

    const size_type N_ = 1 << std::atoi(argv[1]);
    const int steps_between_measurements = 100;

    Diffusion2D System(D, rmax, rmin, N_);

    float time = 0;

    timer runtime;
    runtime.start();

    while(time < tmax){
        System.PropagateDensity(steps_between_measurements);
		cudaDeviceSynchronize();
        time = System.GetTime();
        float moment = System.GetMoment();
        // std::cout << time << '\t' << moment << std::endl;
    }

    runtime.stop();

    double elapsed = runtime.get_timing();

    std::cerr << argv[0] << "\t\tN=" <<N_ << "\t time=" << elapsed << "s" << std::endl;

    std::string density_file = "build/Density_cu_" + std::to_string(N_) + ".dat";
    System.WriteDensity(density_file);

    return 0;
}
