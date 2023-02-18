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

cudaError_t err = cudaSuccess;

#define MAX_NNB	256

__global__ void compute_dist(double *xdata, double *q, int npat, int lpat, double *dist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < npat) {
        double sum = 0.0;
        for (int j = 0; j < lpat; j++) {
            double diff = xdata[i*lpat+j] - q[j];
            sum += diff*diff;
        }
        dist[i] = sum;
    }
}

void compute_knn_brute_force(double *xdata, double *q,double *d_dist, int npat, int lpat, int knn, int *nn_x, double *nn_d, double *dist)
{
	int i, max_i;
	double max_d, new_d;


	// initialize pairs of index and distance 
	for (i = 0; i < knn; i++) {
		nn_x[i] = -1;
		nn_d[i] = 1e99-i;
	}

	// max_d = compute_max_pos(nn_d, knn, &max_i);

    // create array of npat elements with cudaHostAlloc
	// err = cudaHostAlloc((void **)&dist, npat*sizeof(double), cudaHostAllocDefault);
	// if(err != cudaSuccess) {
	// 	fprintf(stderr, "Failed to allocate host vector dist (error code %s)! %d\n", cudaGetErrorString(err), __LINE__);
	// 	exit(EXIT_FAILURE);
	// }

    int threadsPerBlock = 256;
    int blocksPerGrid = (npat + threadsPerBlock - 1) / threadsPerBlock;

    // compute distances to all points
    compute_dist<<<blocksPerGrid, threadsPerBlock>>>(xdata, q, npat, lpat, d_dist);

    cudaDeviceSynchronize();
    cudaMemcpy(dist, d_dist, npat*sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if(err != cudaSuccess) {
        fprintf(stderr, "Failed to launch compute_dist kernel (error code %s)! %d", cudaGetErrorString(err), __LINE__);
        exit(EXIT_FAILURE);
    }

    // compute distances to all points
    // for (i = 0; i < npat; i++) {
    //     dist[i] = compute_dist(q, xdata[i], lpat);
    // }

	for (i = 0; i < npat; i++) {
		new_d = dist[i];	// euclidean
		if (new_d < max_d) {	// add point to the  list of knns, replace element max_i
			max_d = compute_max_pos(nn_d, knn, &max_i);
			nn_x[max_i] = i;
			nn_d[max_i] = new_d;
		}
		
	}

	// sort the knn list 

    int j;
	int temp_x;
	double temp_d;

	for (i = (knn - 1); i > 0; i--) {
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


double find_knn_value(double *d_xmem, double *d_x, double *d_dist,double *p, int n, int knn, double *dist)
{
	int nn_x[MAX_NNB];
	double nn_d[MAX_NNB];

	compute_knn_brute_force(d_xmem, d_x, d_dist, TRAINELEMS, PROBDIM, knn, nn_x, nn_d, dist); // brute-force /linear search

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

    double *d_xmem, *d_x, *d_dist;

    cudaMalloc((void **)&d_xmem, TRAINELEMS*PROBDIM*sizeof(double));
    cudaMalloc((void **)&d_x, QUERYELEMS*PROBDIM*sizeof(double *));
    cudaMalloc((void **)&d_dist, TRAINELEMS*sizeof(double));

    cudaMemcpy(d_xmem, xmem, TRAINELEMS*PROBDIM*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, QUERYELEMS*PROBDIM*sizeof(double), cudaMemcpyHostToDevice);


	double *dist;
	cudaHostAlloc((void **)&dist, TRAINELEMS*sizeof(double), cudaHostAllocDefault);

	printf("Computing KNN...\n");
    

	double t0, t1, t_first = 0.0, t_sum = 0.0;
	double sse = 0.0;
	double err, err_sum = 0.0;

    for (int i=0;i<QUERYELEMS;i++) {	/* requests */

        t0 = gettime();
        double yp = find_knn_value(d_xmem, &d_x[i*PROBDIM], d_dist, &x[i*PROBDIM], PROBDIM, NNBS, dist);
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


	//Free the allocated memory
	// free(xmem);
	// free(xdata);
	// free(ydata);
	// free(x);
	// free(y);

	// cudaFree(d_xmem);
	// cudaFree(d_x);
	// cudaFree(d_dist);

	return 0;
}
