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

#define MAX_NNB 256

void compute_knn_brute_force(double **xdata, double *q, int n_train, int n_dim, int knn, int *nn_x, double *nn_d)
{
    int i, max_i;
    double max_d, new_d;


    // initialize pairs of index and distance 

#pragma acc parallel loop independent present(nn_x[0:knn], nn_d[0:knn])
    for (i = 0; i < knn; i++) {
        nn_x[i] = -1;
        nn_d[i] = 1e99-i;
    }


    max_d = compute_max_pos(nn_d, knn, &max_i);

#pragma acc parallel loop independent present(nn_x[0:knn], nn_d[0:knn], xdata[0:n_train][0:n_dim], q[0:n_dim]) create(max_d) reduction(max:max_d) 
    for (i = 0; i < n_train; i++) {
        new_d = compute_dist(q, xdata[i], n_dim);   // euclidean
        if (new_d < max_d) {    // add point to the  list of knns, replace element max_i
            nn_x[max_i] = i;
            nn_d[max_i] = new_d;
        }
        max_d = compute_max_pos(nn_d, knn, &max_i);
    }

    // sort the knn list 

    int j;
    int temp_x;
    double temp_d;

#pragma acc parallel loop present(nn_d[0:knn], nn_x[0:knn]) create(temp_d, temp_x)
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
double predict_value(int dim, int knn, double *xd, double *yd, double *point, double *dist)
{
    int i;
    double sum_v = 0.0;
    // plain mean (other possible options: inverse distance weight, closest value inheritance)

#pragma acc parallel loop present(yd[0:knn]) reduction(+:sum_v)
    for (i = 0; i < knn; i++) {
        sum_v += yd[i];
    }

    return sum_v/knn;
}

double find_knn_value(double *q, int n_dim, int knn)
{

    int nn_x[MAX_NNB];
    double nn_d[MAX_NNB];
#pragma acc enter data pcreate(nn_x[0:knn], nn_d[0:knn])

    compute_knn_brute_force(xdata, q, TRAINELEMS, PROBDIM, knn, nn_x, nn_d); // brute-force /linear search

    const int dim = PROBDIM;
    const int nd = knn;   // number of points
    double xd[MAX_NNB*PROBDIM];   // points
    double yd[MAX_NNB];     // function values
#pragma acc enter data pcreate(xd[0:NNBS*PROBDIM], yd[0:NNBS])

#pragma acc parallel loop independent present(nn_x[0:knn], ydata[0:TRAINELEMS], yd[0:knn])
    for (int i = 0; i < knn; i++) {
        yd[i] = ydata[nn_x[i]];
    }

#pragma acc parallel loop independent present(nn_x[0:knn], xdata[0:TRAINELEMS][0:PROBDIM], xd[0:nd*dim])
    for (int i = 0; i < knn; i++) {
        for (int j = 0; j < PROBDIM; j++) {
            xd[i*dim+j] = xdata[nn_x[i]][j];
        }
    }

    double fi;

    fi = predict_value(dim, nd, xd, yd, q, nn_d);

// #pragma acc exit data delete(nn_x[0:knn], nn_d[0:knn], xd[0:nd*dim], fd[0:nd])

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

    for (int i=0;i<QUERYELEMS;i++) {

        for (int k = 0; k < PROBDIM; k++)
            x[i * PROBDIM + k] = read_nextnum(fpin);
#if defined(SURROGATES)
        y[i] = read_nextnum(fpin);
#else
        y[i] = 0.0;
#endif
    }

    fclose(fpin);

#pragma acc enter data copyin(xdata[0:TRAINELEMS][0:PROBDIM], ydata[0:TRAINELEMS])
#pragma acc enter data copyin(x[0:QUERYELEMS*PROBDIM], y[0:QUERYELEMS])
    
    //FILE *fpout = fopen("output.knn.txt","w");

    double t0, t1, t_first = 0.0, t_sum = 0.0;
    double sse = 0.0;
    double err, err_sum = 0.0;

    for (int i=0;i<QUERYELEMS;i++) {    /* requests */

        t0 = gettime();
        double yp = find_knn_value(&x[i*PROBDIM], PROBDIM, NNBS);
        t1 = gettime();
        t_sum += (t1-t0);
        if (i == 0) t_first = (t1-t0);

        sse += (y[i]-yp)*(y[i]-yp);

        //for (k = 0; k < PROBDIM; k++)
        //  fprintf(fpout,"%.5f ", x[k]);

        err = 100.0*fabs((yp-y[i])/y[i]);
        //fprintf(fpout,"%.5f %.5f %.2f\n", y[i], yp, err);
        err_sum += err;
    }

#pragma acc exit data delete(xdata[0:TRAINELEMS][0:PROBDIM], ydata[0:TRAINELEMS])
#pragma acc exit data copyout(x[0:QUERYELEMS*PROBDIM], y[0:QUERYELEMS])

    //fclose(fpout);

    double mse = sse/QUERYELEMS;
    double ymean = compute_mean(y, QUERYELEMS);
    double var = compute_var(y, QUERYELEMS, ymean);
    double r2 = 1-(mse/var);

    printf("Results for %d query points\n", QUERYELEMS);
    printf("APE = %.2f %%\n", err_sum/QUERYELEMS);
    printf("MSE = %.6f\n", mse);
    printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

    t_sum = t_sum*1000.0;           // convert to ms
    t_first = t_first*1000.0;   // convert to ms
    printf("Total time = %lf ms\n", t_sum);
    printf("Time for 1st query = %lf ms\n", t_first);
    printf("Time for 2..N queries = %lf ms\n", t_sum-t_first);
    printf("Average time/query = %lf ms\n", (t_sum-t_first)/(QUERYELEMS-1));

    return 0;
}
