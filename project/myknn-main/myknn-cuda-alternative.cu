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

#define MAX_NNB	256

__global__ void compute_dist(double *xdata, double *q, int npat, int lpat, double *dist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < QUERYELEMS && j < TRAINELEMS) {
        double sum = 0.0;
        for (int k = 0; k < PROBDIM; k++) {
            double diff = q[i*PROBDIM+k] - xdata[j*PROBDIM+k];
            sum += diff * diff;
        }
        dist[i*TRAINELEMS+j] = sum;
    }
    if (i == 1 && j == 0) {
        //print q[0] and xdata[0]
        printf("q[i] = %f, xdata[i] = %f\n", q[i*PROBDIM+j], xdata[j*PROBDIM+j]);
        printf("dist[0] = %f\n", dist[i*TRAINELEMS+j]);
    }
}


/* compute an approximation based on the values of the neighbors */
double predict_value(int dim, int knn, double *xdata, double *ydata, double *point, double *dist)
{
	// plain mean (other possible options: inverse distance weight, closest value inheritance)

    /*
    double sum_wv = 0.0;
    double sum_w = 0.0;
    double w;

    for (int i = 0; i < knn; i++) {
        w = 1.0 / (dist[i] + EPSILON);
        sum_wv += w * ydata[i];
        sum_w += w;
    }

    return sum_wv / sum_w;
    */        


	int i;
	double sum_v = 0.0;
	for (i = 0; i < knn; i++) {
		sum_v += ydata[i];
	}

	return sum_v/knn;
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

	double *dist;
    cudaHostAlloc((void **)&dist, QUERYELEMS*TRAINELEMS*sizeof(double), cudaHostAllocDefault);
	int *nn_x = (int *)malloc(QUERYELEMS*MAX_NNB*sizeof(int));
	double *nn_d = (double *)malloc(QUERYELEMS*MAX_NNB*sizeof(double));
	double *y_pred = (double *)malloc(QUERYELEMS*sizeof(double));
	double *sse_arr = (double *)malloc(QUERYELEMS*sizeof(double));
	double *err_arr = (double *)malloc(QUERYELEMS*sizeof(double));

    double *dist_d, *x_d, *xmem_d;
    cudaMalloc((void **)&dist_d, QUERYELEMS*TRAINELEMS*sizeof(double));
    cudaMalloc((void **)&x_d, QUERYELEMS*PROBDIM*sizeof(double));
    cudaMalloc((void **)&xmem_d, TRAINELEMS*PROBDIM*sizeof(double));
    cudaMemcpy(xmem_d, xmem, TRAINELEMS*PROBDIM*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x, QUERYELEMS*PROBDIM*sizeof(double), cudaMemcpyHostToDevice);


	double t0, t1, t_first = 0.0, t_sum = 0.0;
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((QUERYELEMS + threadsPerBlock.x - 1) / threadsPerBlock.x, (TRAINELEMS + threadsPerBlock.y - 1) / threadsPerBlock.y);
	t0 = gettime();

    compute_dist<<<numBlocks, threadsPerBlock>>>(xmem_d, x_d, TRAINELEMS, PROBDIM, dist_d);
    cudaMemcpy(dist, dist_d, QUERYELEMS*TRAINELEMS*sizeof(double), cudaMemcpyDeviceToHost); 

	for(int i = 0; i < QUERYELEMS; i++) {
		for(int j = 0; j < MAX_NNB; j++) {
			nn_x[i*MAX_NNB+j] = -1;
			nn_d[i*MAX_NNB+j] = 1e99-i;
		}
	}

	for (int i = 0; i < QUERYELEMS; i++) {
		int max_i;
		double max_d, new_d;
		int knn = NNBS;
		int lpat = PROBDIM;

		
		max_d = compute_max_pos(&nn_d[i*MAX_NNB], knn, &max_i);

		for (int j = 0; j < TRAINELEMS; j++) {
			new_d = dist[i*TRAINELEMS+j];	// euclidean
			if (new_d < max_d) {	// add point to the  list of knns, replace element max_i
				nn_x[i*MAX_NNB + max_i] = j;
				nn_d[i*MAX_NNB + max_i] = new_d;
				max_d = compute_max_pos(&nn_d[i*MAX_NNB], knn, &max_i);
			}
		}

		// sort the knn list 
		quicksort(&nn_d[i*MAX_NNB], &nn_x[i*MAX_NNB], 0, knn-1);
	}
	

	// // compute the predicted values
	for (int i = 0; i < QUERYELEMS; i++) {
		int knn = NNBS;
		int dim = PROBDIM;
		int nd = knn;   // number of points
		double xd[MAX_NNB*PROBDIM];   // points
		double fd[MAX_NNB];     // function values

		for (int j = 0; j < knn; j++) {
			fd[j] = ydata[nn_x[i*MAX_NNB+j]];
		}

		for (int j = 0; j < knn; j++) {
			for (int k = 0; k < PROBDIM; k++) {
				xd[j*dim+k] = xdata[nn_x[i*MAX_NNB + j]][k];
			}
		}

		double fi;

		fi = predict_value(dim, nd, xd, fd, &x[i*PROBDIM], &nn_d[i*MAX_NNB]);

		y_pred[i] = fi;
	}

	t1 = gettime();
	t_sum += (t1-t0);

	// // compute the error
	for (int i = 0; i < QUERYELEMS; i++) {
		sse_arr[i] = (y[i]-y_pred[i])*(y[i]-y_pred[i]);
		err_arr[i] = 100.0*fabs((y[i]-y_pred[i])/y[i]);
	}

	// // compute the average error
	double sse = 0.0;
	double err_sum = 0.0;
	for (int i = 0; i < QUERYELEMS; i++) {
		sse += sse_arr[i];
		err_sum += err_arr[i];
	}

	double mse = sse/QUERYELEMS;
	double ymean = compute_mean(y, QUERYELEMS);
	double var = compute_var(y, QUERYELEMS, ymean);
	double r2 = 1-(mse/var);

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum/QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

	t_sum = t_sum*1000.0;			// convert to ms
	printf("Total time = %lf ms\n", t_sum);
	printf("Average time/query = %lf ms\n", (t_sum)/(QUERYELEMS));

	return 0;
}