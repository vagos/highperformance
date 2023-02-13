#ifndef FUNC_H
#define FUNC_H

#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/* I/O routines */
FILE *open_traindata(char *trainfile)
{
	FILE *fp;

	fp = fopen(trainfile, "r");
	if (fp == NULL) {
		printf("traindata; File %s not available\n", trainfile);
		exit(1);
	}
	return fp;
}

FILE *open_querydata(char *queryfile)
{
	FILE *fp;

	fp = fopen(queryfile, "r");
	if (fp == NULL) {
		printf("querydata: File %s not available\n", queryfile);
		exit(1);
	}
	return fp;
}

double read_nextnum(FILE *fp)
{
	double val;

	int c = fscanf(fp, "%lf", &val);
	if (c <= 0) {
		fprintf(stderr, "fscanf returned %d\n", c);
		exit(1);
	}
	return val;
}

/* Timer */
double gettime()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (double) (tv.tv_sec+tv.tv_usec/1000000.0);
}

/* Function to approximate */
double fitfun(double *x, int n)
{
	double f = 0.0;
	int i;

#if 1
	for(i=0; i<n; i++)	/* circle */
		f += x[i]*x[i];
#endif
#if 0
	for(i=0; i<n-1; i++) {	/*  himmelblau */
		f = f + pow((x[i]*x[i]+x[i+1]-11.0),2) + pow((x[i]+x[i+1]*x[i+1]-7.0),2);
	}
#endif
#if 0
	for (i=0; i<n-1; i++)   /* rosenbrock */
		f = f + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);
#endif
#if 0
	for (i=0; i<n; i++)     /* rastrigin */
		f = f + pow(x[i],2) + 10.0 - 10.0*cos(2*M_PI*x[i]);
#endif

	return f;
}


/* random number generator  */
#define SEED_RAND()     srand48(1)
#define URAND()         drand48()

#ifndef LB
#define LB -1.0
#endif
#ifndef UB
#define UB 1.0
#endif

double get_rand(int k)
{
	return (UB-LB)*URAND()+LB;
}


/* utils */
double compute_min(double *v, int n)
{
	int i;
	double vmin = v[0];
	for (i = 1; i < n; i++)
		if (v[i] < vmin) vmin = v[i];

	return vmin;
}

double compute_max(double *v, int n)
{
	int i;
	double vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) vmax = v[i];

	return vmax;
}

double compute_sum(double *v, int n)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s;
}

double compute_sum_pow(double *v, int n, int p)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i], p);

	return s;
}

double compute_mean(double *v, int n)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s/n;
}

double compute_std(double *v, int n, double mean)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return sqrt(s/(n-1));
}

double compute_var(double *v, int n, double mean)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return s/n;
}

double compute_dist(double *v, double *w, int n)
{
	int i;
	double s = 0.0;
	for (i = 0; i < n; i++) {
		s+= pow(v[i]-w[i],2);
	}

	// return sqrt(s); /* Loses precision but still okay */
	return s;
}

double compute_max_pos(double *v, int n, int *pos)
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

double compute_min_pos(double *v, int n, int *pos)
{
	int i, p = 0;
	double vmin = v[0];
	for (i = 1; i < n; i++)
		if (v[i] < vmin) {
			vmin = v[i];
			p = i;
		}

	*pos = p;
	return vmin;
}

double compute_root(double dist, int norm)
{
	if (dist == 0) return 0;

	switch (norm) {
	case 2:
		return sqrt(dist);
	case 1:
	case 0:
		return dist;
	default:
		return pow(dist, 1 / (double) norm);
	}
}

double compute_distance(double *pat1, double *pat2, int lpat, int norm)
{
	int i;
	double dist = 0.0;

	for (i = 0; i < lpat; i++) {
		double diff = 0.0;

		diff = pat1[i] - pat2[i];

		switch (norm) {
		double   adiff;

		case 2:
			dist += diff * diff;
			break;
		case 1:
			dist += fabs(diff);
			break;
		case 0:
			if ((adiff = fabs(diff)) > dist)
			dist = adiff;
			break;
		default:
			dist += pow(fabs(diff), (double) norm);
			break;
		}
	}

	return dist;	// compute_root(dist);
}

/* 

   ==== Tried out merge sort. Results were actually slower.

void merge(int start, int mid, int end, double* nn_d, int* nn_x) 
{
    int i = start, j = mid + 1, k = 0;
    int temp_x[end - start + 1];
    double temp_d[end - start + 1];

    while (i <= mid && j <= end) {
        if (nn_d[i] < nn_d[j]) {
            temp_d[k] = nn_d[i];
            temp_x[k++] = nn_x[i++];
        } else {
            temp_d[k] = nn_d[j];
            temp_x[k++] = nn_x[j++];
        }
    }
    while (i <= mid) {
        temp_d[k] = nn_d[i];
        temp_x[k++] = nn_x[i++];
    }
    while (j <= end) {
        temp_d[k] = nn_d[j];
        temp_x[k++] = nn_x[j++];
    }
    for (int l = 0; l < k; l++) {
        nn_d[start + l] = temp_d[l];
        nn_x[start + l] = temp_x[l];
    }
}

void merge_sort(int start, int end, double* nn_d, int* nn_x)
{
    if (start < end) {
        int mid = (start + end) / 2;
        #pragma omp task shared(nn_x, nn_d)
        {
            merge_sort(start, mid, nn_d, nn_x);
        }
        #pragma omp task shared(nn_x, nn_d)
        {
            merge_sort(mid + 1, end, nn_d, nn_x);
        }
        #pragma omp taskwait
        merge(start, mid, end, nn_d, nn_x);
    }
}

==== Tried using KD-Trees for nearest neighbor search. Results were actually slower.

void compute_knn_kdtree(double **xdata, double *q, int npat, int lpat, int knn, int *nn_x, double *nn_d, kdt::KDTree<Point> &tree)
{

    const Point query = Point(q);
    const std::vector<int> knnIndices = tree.knnSearch(query, NNBS);

    for (std::size_t i = 0; i < knnIndices.size(); i++)
    {
        nn_x[i] = knnIndices[i];
        nn_d[i] = compute_dist(q, xdata[knnIndices[i]], lpat);
    }

}

*/


#endif /* FUNC_H_ */
