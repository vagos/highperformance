#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#ifndef PROBDIM
#define PROBDIM 2
#endif

#include "func.h"

static double *xdata;
static double ydata[TRAINELEMS];

#define MAX_NNB	256

void compute_knn_brute_force(double *xdata, double *q, int npat, int lpat, int knn, int *nn_x, double *nn_d)
{
	int i, max_i;
	double max_d, new_d;


	// initialize pairs of index and distance 
	for (i = 0; i < NNBS; i++) {
		nn_x[i] = -1;
		nn_d[i] = 1e99-i;
	}

	max_d = compute_max_pos(nn_d, knn, &max_i);

	for (i = 0; i < npat; i++) {
		new_d = compute_dist(q, &xdata[i*PROBDIM], lpat);
		if (new_d < max_d) {	// add point to the  list of knns, replace element max_i
			nn_x[max_i] = i;
			nn_d[max_i] = new_d;
			max_d = compute_max_pos(nn_d, knn, &max_i);
		}
	}

	return;
}


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


double find_knn_value(double *p, int n, int knn)
{
	int nn_x[MAX_NNB];
	double nn_d[MAX_NNB];

	compute_knn_brute_force(xdata, p, TRAINELEMS, PROBDIM, knn, nn_x, nn_d); // brute-force /linear search

	int dim = PROBDIM;
	int nd = knn;   // number of points
	double xd[MAX_NNB*PROBDIM];   // points
	double fd[MAX_NNB];     // function values

	for (int i = 0; i < knn; i++) {
		fd[i] = ydata[nn_x[i]];
	}

	for (int i = 0; i < knn; i++) {
		for (int j = 0; j < PROBDIM; j++) {
			xd[i*dim+j] = xdata[nn_x[i]*PROBDIM + j];
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

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *y;
    double *x;

    if (rank == 0) {

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	xdata = (double *)malloc(TRAINELEMS*PROBDIM*sizeof(double *));

	FILE *fpin = open_traindata(trainfile);

	for (int i=0;i<TRAINELEMS;i++) {
		for (int k = 0; k < PROBDIM; k++)
            xdata[i*PROBDIM + k] = read_nextnum(fpin);

#if defined(SURROGATES)
        ydata[i] = read_nextnum(fpin);
#else
        ydata[i] = 0;
#endif
	}
	fclose(fpin);

    /* Read query data */

	fpin = open_querydata(queryfile);

	y = (double *)malloc(QUERYELEMS*sizeof(double));
	x = (double *)malloc(QUERYELEMS*PROBDIM*sizeof(double));

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

        MPI_Request req[size];

        for (int i = 1; i < size; i++) {

            MPI_Isend(x, QUERYELEMS * PROBDIM, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req[i]);
            MPI_Isend(y, QUERYELEMS, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req[i]);
            MPI_Isend(xdata, TRAINELEMS * PROBDIM, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req[i]);
            MPI_Isend(ydata, TRAINELEMS, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req[i]);
        }

        printf("Data read\n");
    }
    else
    {
        xdata = (double *)malloc(TRAINELEMS*PROBDIM*sizeof(double *));
        y = (double *)malloc(QUERYELEMS*sizeof(double));
        x = (double *)malloc(QUERYELEMS*PROBDIM*sizeof(double));

        MPI_Recv(x, QUERYELEMS * PROBDIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(y, QUERYELEMS, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(xdata, TRAINELEMS * PROBDIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(ydata, TRAINELEMS, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


	double t0, t1, t_first = 0.0, t_sum = 0.0, t_total_0 = 0.0, t_total_1 = 0.0;;
	double sse = 0.0;
	double err, err_sum = 0.0;

    t_total_0 = gettime();
	
    //FILE *fpout = fopen("output.knn.txt","w");

    // Start and end for each process
    int start = rank * (QUERYELEMS / size);
    int end = (rank + 1) * (QUERYELEMS / size);

    for (int i = start; i < end; i++) {	/* requests */

        t0 = gettime();
        double yp = find_knn_value(&x[i*PROBDIM], PROBDIM, NNBS);
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

    // reduction sum to rank 0 for error,sse and max time
    double err_sum_global, sse_global, t_sum_global;
    MPI_Reduce(&err_sum, &err_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sse, &sse_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_sum, &t_sum_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    t_total_1 = gettime();

    if (rank == 0) {
        err_sum = err_sum_global;
        t_sum = t_sum_global;
        sse = sse_global;

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
        printf("Total time with t_sum = %lf ms\n", t_sum);
        printf("Total time with t_total = %lf ms\n", (t_total_1-t_total_0)*1000.0);
        printf("Time for 1st query = %lf ms\n", t_first);
        printf("Time for 2..N queries = %lf ms\n", t_sum-t_first);
        printf("Average time/query = %lf ms\n", (t_sum-t_first)/(QUERYELEMS-1));
    }

    MPI_Finalize();

	return 0;
}
