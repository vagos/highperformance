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


/* compute an approximation based on the values of the neighbors */
double predict_value(int dim, int knn, double *xdata, double *ydata, double *point, double *dist)
{
	// plain mean (other possible options: inverse distance weight, closest value inheritance)

    double sum_wv = 0.0;
    double sum_w = 0.0;
    double w;

    for (int i = 0; i < knn; i++) {
        w = 1.0 / (dist[i] + EPSILON);
        sum_wv += w * ydata[i];
        sum_w += w;
    }

    return sum_wv / sum_w;
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

	double *dist = (double *)malloc(QUERYELEMS*TRAINELEMS*sizeof(double));
	int *nn_x = (int *)malloc(QUERYELEMS*MAX_NNB*sizeof(int));
	double *nn_d = (double *)malloc(QUERYELEMS*MAX_NNB*sizeof(double));
	double *y_pred = (double *)malloc(QUERYELEMS*sizeof(double));
	double *sse_arr = (double *)malloc(QUERYELEMS*sizeof(double));
	double *err_arr = (double *)malloc(QUERYELEMS*sizeof(double));

	double t0, t1,t_sum = 0.0;
	

#pragma acc data copyin(xdata[0:TRAINELEMS][0:PROBDIM], ydata[0:TRAINELEMS]) copyin(x[0:QUERYELEMS*PROBDIM], y[0:QUERYELEMS]) create(dist[0:QUERYELEMS*TRAINELEMS], nn_x[0:QUERYELEMS*MAX_NNB], nn_d[0:QUERYELEMS*MAX_NNB]) copyout(y_pred[0:QUERYELEMS])
{
	t0 = gettime();

	// compute distances
#pragma acc kernels loop independent gang vector collapse(2) present(dist[0:QUERYELEMS*TRAINELEMS])
	for (int i = 0; i < QUERYELEMS; i++) {
		for (int j = 0; j < TRAINELEMS; j++) {
			dist[i*TRAINELEMS+j] = compute_dist(&x[i*PROBDIM], xdata[j], PROBDIM);
		}
	}

//copyout dist[0:QUERYELEMS*TRAINELEMS]

#pragma acc kernels loop independent gang vector collapse(2) present(nn_x[0:QUERYELEMS*MAX_NNB], nn_d[0:QUERYELEMS*MAX_NNB])
	for(int i = 0; i < QUERYELEMS; i++) {
		for(int j = 0; j < NNBS; j++) {
			nn_x[i*MAX_NNB+j] = -1;
			nn_d[i*MAX_NNB+j] = 1e99-i;
		}
	}

    int max_i;
    double max_d, new_d;
    int knn = NNBS;

#pragma acc kernels loop independent private(max_i, max_d, new_d, knn) present(nn_x[0:QUERYELEMS*MAX_NNB], nn_d[0:QUERYELEMS*MAX_NNB], dist[0:QUERYELEMS*TRAINELEMS])
	for (int i = 0; i < QUERYELEMS; i++) {

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
		// quicksort(&nn_d[i*MAX_NNB], &nn_x[i*MAX_NNB], 0, knn-1);
	}
	
	// compute the predicted values
#pragma acc kernels loop independent present(y_pred[0:QUERYELEMS], nn_x[0:QUERYELEMS*MAX_NNB], nn_d[0:QUERYELEMS*MAX_NNB], ydata[0:TRAINELEMS])
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
