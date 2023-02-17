#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifndef PROBDIM
#define PROBDIM 2
#endif

#include "func.h"

static double **xdata;
static double ydata[TRAINELEMS];

// static const int diffusion_block_x = 16;
// static const int diffusion_block_y = 16;
cudaError_t err = cudaSuccess;

#define MAX_NNB	256

__device__ double compute_max_pos_d(double *v, int n, int *pos)
{
	int i, p = 0;
	double vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) {
			vmax = v[i];
			p = i;
		}

	*pos = p;
	return vmax;
}

__device__ double compute_dist_d(double *v, double *w, int n)
{
	int i;
	double s = 0.0;
	for (i = 0; i < n; i++) {
		s+= pow(v[i]-w[i],2);
	}

	// return sqrt(s); /* Loses precision but still okay */
	return s;
}

__global__ void compute_knn_brute_force_kernel(double *xmem, double *q, int npat, int lpat, int knn, int *nn_x, double *nn_d)
{
    int i, max_i;
    double max_d, new_d;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    // // initialize pairs of index and distance
    // if (i < knn) {
    //     nn_x[i] = -1;
    //     nn_d[i] = 1e99-i;
    // }

    max_d = compute_max_pos_d(nn_d, knn, &max_i);

    if (i < npat) {
        new_d = compute_dist_d(q, xmem + i * PROBDIM, lpat);	// euclidean
        if (new_d < max_d) {	// add point to the  list of knns, replace element max_i
            nn_x[max_i] = i;
            nn_d[max_i] = new_d;
        }
        max_d = compute_max_pos_d(nn_d, knn, &max_i);
    }
}





void compute_knn_brute_force(double *d_xmem, double *d_q, int npat, int lpat, int knn,int *d_nn_x,double *d_nn_d, int *nn_x, double *nn_d)
{
    // double *d_xmem, *d_q, *d_nn_d;
    // int *d_nn_x;
    // err = cudaMalloc((void **)&d_xmem, TRAINELEMS*PROBDIM*sizeof(double));
    // err = cudaMalloc((void **)&d_q, PROBDIM*sizeof(double));
    // err = cudaMalloc((void **)&d_nn_d, MAX_NNB*sizeof(double));
    // err = cudaMalloc((void **)&d_nn_x, MAX_NNB*sizeof(int));

    // err = cudaMemcpy(d_xmem, xdata, TRAINELEMS*PROBDIM*sizeof(double), cudaMemcpyHostToDevice);
    // err = cudaMemcpy(d_q, q, PROBDIM*sizeof(double), cudaMemcpyHostToDevice);

    // //define grid_size and block_size
    // int threads_x = diffusion_block_x;
    // int threads_y = diffusion_block_y;
    // dim3 threadsPerBlock(threads_x,threads_y);

    // int blocks_x = ( + threads_x - 1) / threads_x;
    // int blocks_y = (npat + threads_y - 1) / threads_y;
    // blocks_x = (blocks_x == 0)? 1 : blocks_x;
    // blocks_y = (blocks_y == 0)? 1 : blocks_y;

    // dim3 blocksPerGrid(4,4);

    // compute_knn_brute_force_kernel<<<256,512>>>(d_xmem, d_q, npat, lpat, knn, d_nn_x, d_nn_d);


	// int i, max_i;
	// double max_d, new_d;


	// initialize pairs of index and distance 
	// for (i = 0; i < knn; i++) {
	// 	nn_x[i] = -1;
	// 	nn_d[i] = 1e99-i;
	// }

	err = cudaMemset(d_nn_x, -1, MAX_NNB*sizeof(int));
	err = cudaMemset(d_nn_d, 0x7F, MAX_NNB*sizeof(double));

	// err = cudaMemcpy(d_nn_x, nn_x, MAX_NNB*sizeof(int), cudaMemcpyHostToDevice);
	// err = cudaMemcpy(d_nn_d, nn_d, MAX_NNB*sizeof(double), cudaMemcpyHostToDevice);

    // err = cudaMemcpy(nn_x, d_nn_x, MAX_NNB*sizeof(int), cudaMemcpyDeviceToHost);
    // err = cudaMemcpy(nn_d, d_nn_d, MAX_NNB*sizeof(double), cudaMemcpyDeviceToHost);

    compute_knn_brute_force_kernel<<<32,64>>>(d_xmem, d_q, npat, lpat, knn, d_nn_x, d_nn_d);

	// max_d = compute_max_pos(nn_d, knn, &max_i);

	// for (i = 0; i < npat; i++) {
	// 	new_d = compute_dist(q, xdata[i], lpat);	// euclidean
	// 	if (new_d < max_d) {	// add point to the  list of knns, replace element max_i
	// 		nn_x[max_i] = i;
	// 		nn_d[max_i] = new_d;
	// 	}
	// 	max_d = compute_max_pos(nn_d, knn, &max_i);
	// }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s in %s at line %d in file %s \n", cudaGetErrorString(err), __FUNCTION__, __LINE__, __FILE__);
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(nn_x, d_nn_x, MAX_NNB*sizeof(int), cudaMemcpyDeviceToHost);
    err = cudaMemcpy(nn_d, d_nn_d, MAX_NNB*sizeof(double), cudaMemcpyDeviceToHost);


	// sort the knn list 

    int j;
	int temp_x;
	double temp_d;

	for (int i = (knn - 1); i > 0; i--) {
		for (j = 1; j <= i; j++) {
			if (nn_d[j-1] > nn_d[j]) {
				temp_d = nn_d[j-1]; nn_d[j-1] = nn_d[j]; nn_d[j] = temp_d;
				temp_x = nn_x[j-1]; nn_x[j-1] = nn_x[j]; nn_x[j] = temp_x;
			}
		}
	}

	return;
}


/* compute an approximation based on the values of the neighbors */
double predict_value(int dim, int knn, double *xdata, double *ydata, double *point, double *dist)
{
	int i;
	double sum_v = 0.0;
	// plain mean (other possible options: inverse distance weight, closest value inheritance)

	for (i = 0; i < knn; i++) {
		sum_v += ydata[i];
	}

	return sum_v/knn;
}


double find_knn_value(double *d_xmem,double *d_q, int *d_nn_x, double *d_nn_d, double *p, int n, int knn)
{
	int nn_x[MAX_NNB];
	double nn_d[MAX_NNB];

	compute_knn_brute_force(d_xmem, d_q, TRAINELEMS, PROBDIM, knn, d_nn_x, d_nn_d, nn_x, nn_d); // brute-force /linear search

	err = cudaMemcpy(nn_x, d_nn_x, MAX_NNB*sizeof(int), cudaMemcpyDeviceToHost);
	err = cudaMemcpy(nn_d, d_nn_d, MAX_NNB*sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s in %s at line %d in file %s \n", cudaGetErrorString(err), __FUNCTION__, __LINE__, __FILE__);
		exit(EXIT_FAILURE);
	}

	int dim = PROBDIM;
	int nd = knn;   // number of points
	double xd[MAX_NNB*PROBDIM];   // points
	double fd[MAX_NNB];     // function values

	for (int i = 0; i < knn; i++) {
		fd[i] = ydata[nn_x[i]];
	}

	for (int i = 0; i < knn; i++) {
		for (int j = 0; j < PROBDIM; j++) {
			xd[i*dim+j] = xdata[nn_x[i]][j];
		}
	}

	double fi;

	fi = predict_value(dim, nd, xd, fd, p, nn_d);

	return fi;
}

int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	double *xmem = (double *)malloc(TRAINELEMS*PROBDIM*sizeof(double));
	xdata = (double **)malloc(TRAINELEMS*sizeof(double *));
	for (int i = 0; i < TRAINELEMS; i++) xdata[i] = xmem + i*PROBDIM; //&xmem[i*PROBDIM];

	FILE *fpin = open_traindata(trainfile);

	for (int i=0;i<TRAINELEMS;i++) {
		for (int k = 0; k < PROBDIM; k++)
            xdata[i][k] = read_nextnum(fpin);

#if defined(SURROGATES)
        ydata[i] = read_nextnum(fpin);
#else
        ydata[i] = 0;
#endif
	}
	fclose(fpin);

    /* Read query data */

	fpin = open_querydata(queryfile);

	double *y = (double *)malloc(QUERYELEMS*sizeof(double));
	double *x = (double *)malloc(QUERYELEMS*PROBDIM*sizeof(double));

	for (int i=0;i<QUERYELEMS;i++) {	/* requests */

        for (int k = 0; k < PROBDIM; k++)
            x[i * PROBDIM + k] = read_nextnum(fpin);
#if defined(SURROGATES)
		y[i] = read_nextnum(fpin);
#else
		y[i] = 0.0;
#endif
    }

	fclose(fpin);
	
    //FILE *fpout = fopen("output.knn.txt","w");

	double *d_xmem, *d_q, *d_nn_d;
    int *d_nn_x;
    err = cudaMalloc((void **)&d_xmem, TRAINELEMS*PROBDIM*sizeof(double));
    err = cudaMalloc((void **)&d_q, QUERYELEMS*PROBDIM*sizeof(double));
    err = cudaMalloc((void **)&d_nn_d, MAX_NNB*sizeof(double));
    err = cudaMalloc((void **)&d_nn_x, MAX_NNB*sizeof(int));

	err = cudaMemcpy(d_xmem, xmem, TRAINELEMS*PROBDIM*sizeof(double), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_q, x, QUERYELEMS*PROBDIM*sizeof(double), cudaMemcpyHostToDevice);

	double t0, t1, t_first = 0.0, t_sum = 0.0;
	double sse = 0.0;
	double err, err_sum = 0.0;

    for (int i=0;i<QUERYELEMS;i++) {	/* requests */

        t0 = gettime();
        double yp = find_knn_value(d_xmem,&d_q[i*PROBDIM],d_nn_x, d_nn_d, &x[i*PROBDIM], PROBDIM, NNBS);
        t1 = gettime();
        t_sum += (t1-t0);
        if (i == 0) t_first = (t1-t0);

        sse += (y[i]-yp)*(y[i]-yp);

        //for (k = 0; k < PROBDIM; k++)
        //	fprintf(fpout,"%.5f ", x[k]);

        err = 100.0*fabs((yp-y[i])/y[i]);
        //fprintf(fpout,"%.5f %.5f %.2f\n", y[i], yp, err);
        err_sum += err;
	}

	//fclose(fpout);

	double mse = sse/QUERYELEMS;
	double ymean = compute_mean(y, QUERYELEMS);
	double var = compute_var(y, QUERYELEMS, ymean);
	double r2 = 1-(mse/var);

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum/QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

	t_sum = t_sum*1000.0;			// convert to ms
	t_first = t_first*1000.0;	// convert to ms
	printf("Total time = %lf ms\n", t_sum);
	printf("Time for 1st query = %lf ms\n", t_first);
	printf("Time for 2..N queries = %lf ms\n", t_sum-t_first);
	printf("Average time/query = %lf ms\n", (t_sum-t_first)/(QUERYELEMS-1));

	return 0;
}
