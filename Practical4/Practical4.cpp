#include "stdio.h"
#include "omp.h"
#define NUM_THREADS 16
static long num_steps = 100000;
double step;

const long MAX = 100000;





void P4Q1b() {
    int i, nthreads;
    double pi = 0;
    double sum = 0.0;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);
    double start_time = omp_get_wtime();

#pragma omp parallel
    {
        int i, id, nthrds;
        double x, partial_sum;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();

        if (id == 0) {
            nthreads = nthrds;
            printf("Total OMP threads: %d\n", nthreads);
        }

        partial_sum = 0.0;

        for (i = id; i < num_steps; i = i + nthrds)
        {
            x = (i + 0.5) * step;
            partial_sum += 4.0 / (1.0 + x * x);
        }

#pragma omp critical
        {
            sum += partial_sum;
        }


    }
    pi = sum * step;
    double end_time = omp_get_wtime();

    printf("%f\n", pi);

    printf("Improve Work took %f seconds\n", end_time - start_time);
}

void P4Q1a() {
    int i, nthreads;
    double pi = 0;
    double partial_sums[NUM_THREADS], sum = 0.0;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);
    double start_time = omp_get_wtime();
#pragma omp parallel
    {
        int i, id, nthrds;
        double x;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        if (id == 0) {
            nthreads = nthrds;
            printf("Total OMP threads: %d\n", nthreads);
        }

        for (i = id, partial_sums[id] = 0.0; i < num_steps; i = i + nthrds)
        {
            x = (i + 0.5) * step;
            partial_sums[id] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < nthreads; i++) {
        sum += partial_sums[i];
    }
    pi = sum * step;
    double end_time = omp_get_wtime();

    printf("%f\n", pi);

    printf("Ori Work took %f seconds\n", end_time - start_time);
}

void P4Q2a() {
    double ave = 0.0, A[MAX];
    int i;
    for (i = 0; i < MAX; i++)
    {
        A[i] = i;
    }
    double start_time = omp_get_wtime();

    for (i = 0; i < MAX; i++) {
        ave += A[i];
    }
    double end_time = omp_get_wtime();
    ave = ave / MAX;
    printf("%f\n", ave);
    printf("Ori Work took %f seconds\n", end_time - start_time);

}

void P4Q2bb() {
    double ave = 0.0, A[MAX];
    int i;
    double sum = 0.0;
    omp_set_num_threads(NUM_THREADS);
    for (i = 0; i < MAX; i++)
    {
        A[i] = i;
    }
    double start_time = omp_get_wtime();

    for (i = 0; i < MAX; i++) {
        ave += A[i];
    }
    double end_time = omp_get_wtime();
    ave = ave / MAX;
    printf("%f\n", ave);
    printf("Ori Work took %f seconds\n", end_time - start_time);

}

void P4Q2b() {
    double ave = 0.0, A[MAX];
    int i;
    for (i = 0; i < MAX; i++)
    {
        A[i] = i;
    }
    double start_time = omp_get_wtime();

#pragma omp parallel for num_threads(16)
    for (i = 0; i < MAX; i++) {
#pragma omp critical
        {
            ave += A[i];
        }
    }

    double end_time = omp_get_wtime();
    ave = ave / MAX;
    printf("%f\n", ave);
    printf("Second method Work took %f seconds\n", end_time - start_time);
}

void P4Q2c() {
    double ave = 0.0, A[MAX];
    int i;

    //initialize A
    for (i = 0; i < MAX; i++)
    {
        A[i] = i;
    }

    double start_time = omp_get_wtime();

#pragma omp parallel for reduction(+:ave)
    for (i = 0; i < MAX; i++) {
        ave += A[i];
    }

    double end_time = omp_get_wtime();
    ave = ave / MAX;

    printf("%f\n", ave);
    printf("Third method Work took %f seconds\n", end_time - start_time);
}

int main()
{
    //P4Q1a();
    //P4Q1a();
    //P4Q1a();
    //P4Q1a();
    //P4Q1b();
    //P4Q1b();
    //P4Q1b();
    //P4Q1b();

    P4Q2a();
    P4Q2a();
    P4Q2a();
    P4Q2a();

    P4Q2b();
    P4Q2b();
    P4Q2b();
    P4Q2b();

    P4Q2c();
    P4Q2c();
    P4Q2c();
    P4Q2c();
    return 0;
}