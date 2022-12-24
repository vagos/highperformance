#include <cmath>
#include <immintrin.h>
#include <x86intrin.h>
#include <omp.h>

#include "particles.h"
#include "utils.h"

inline double hsum(const __m256d val)
{
  double r = val[0] + val[1] + val[2] + val[3];

  return r;
}

/**
 * Compute the gravitational forces in the system of particles
 * Use AVX and OpenMP to speed up computation
**/
void computeGravitationalForcesFast(Particles& particles)
{
    // std::cout << "COMPUTING..." << std::endl;

	const double G = 6.67408e-11;

#pragma omp parallel for
    for (int i=0; i<particles.n; i++)
    {

       particles.fx[i] = 0.0f;
       particles.fy[i] = 0.0f;
       particles.fz[i] = 0.0f;

      __m256d p_x = _mm256_set1_pd(particles.x[i]);
      __m256d p_y = _mm256_set1_pd(particles.y[i]);
      __m256d p_z = _mm256_set1_pd(particles.z[i]);

      __m256d p_m = _mm256_set1_pd(particles.m[i]);

       for (int j=0; j+3 <= particles.n; j+=4)
       {

           __m256d p_m_j = _mm256_loadu_pd(particles.m + j);

           __m256d p_x_j = _mm256_loadu_pd(particles.x + j);
           __m256d p_y_j = _mm256_loadu_pd(&particles.y[j]);
           __m256d p_z_j = _mm256_loadu_pd(&particles.z[j]);

           __m256d sub_x = _mm256_sub_pd(p_x, p_x_j);
           __m256d sub_y = _mm256_sub_pd(p_y, p_y_j);
           __m256d sub_z = _mm256_sub_pd(p_z, p_z_j);

           __m256d sub_x_2 = _mm256_mul_pd(sub_x, sub_x);
           __m256d sub_y_2 = _mm256_mul_pd(sub_y, sub_y);
           __m256d sub_z_2 = _mm256_mul_pd(sub_z, sub_z);

           __m256d r_len_2 = _mm256_add_pd(sub_x_2, sub_y_2);
           r_len_2 = _mm256_add_pd(r_len_2, sub_z_2);

           __m256d r_len = _mm256_sqrt_pd(r_len_2);
           __m256d r_len_3 = _mm256_mul_pd(r_len, r_len_2);

           __m256d mm = _mm256_mul_pd(p_m, p_m_j);
           __m256d GGGG = _mm256_set1_pd(G);

           __m256d gmm = _mm256_mul_pd(mm, GGGG);

           __m256d magnitude = _mm256_div_pd(gmm, r_len_3);

           if (j <= i && i < j+4)      {
              const __m256d iiii     = _mm256_set1_pd(i);
              const __m256d j0j1j2j3 = _mm256_set_pd(j+3.0, j+2.0, j+1.0, j);
              const __m256d mask = _mm256_cmp_pd(iiii, j0j1j2j3, _CMP_NEQ_OQ);
              magnitude = _mm256_and_pd(magnitude, mask);
          }
          
          __m256d f_x = _mm256_mul_pd(sub_x, magnitude);
          __m256d f_y = _mm256_mul_pd(sub_y, magnitude);
          __m256d f_z = _mm256_mul_pd(sub_z, magnitude);

          particles.fx[i] += hsum(f_x);
          particles.fy[i] += hsum(f_y);
          particles.fz[i] += hsum(f_z);
       }
       
    }
}

int main()
{
	testAll(computeGravitationalForcesFast, false);
	return 0;
}
