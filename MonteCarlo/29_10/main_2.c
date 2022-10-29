
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

int main(int argc, char** argv)
{
    int rank, size, flag = 0, total_N = 0, bufElems = 0;
    int *displs, *scounts;
    const int N = 100000;
    double eps = strtold(argv[1], NULL), glob_sum = 0, sum = 0, acc_sum = 0;
    double diff_time = 0, max_time, gen_time = 0;
    const double real_val = 1.0 / 24;
    MPI_Status status_x, status_y, status_z, status;
    MPI_Init(&argc, &argv);
    srand48(57);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double *x, *y, *z, * x_sl, *y_sl, *z_sl;
    if (!rank)
    {
        x = malloc(N * sizeof(double));
        y = malloc(N * sizeof(double));
        z = malloc(N * sizeof(double));
    }
    else
    {
        bufElems = (rank != size - 1) ? N / (size - 1) : N - (N / (size - 1)) * (size - 2);
    }
    x_sl = malloc(bufElems * sizeof(double));
    y_sl = malloc(bufElems * sizeof(double));
    z_sl = malloc(bufElems * sizeof(double));    
    displs = (int*)malloc(size * sizeof(int));
    scounts = (int*)malloc(size * sizeof(int));
    displs[0] = 0;
    scounts[0] = 0;
    for (int i = 1; i < size - 1; ++i)
    {
        displs[i] = (N / (size - 1)) * (i - 1);
        scounts[i] = N / (size - 1);
    }
    displs[size - 1] = (N / (size - 1)) * (size - 2);
    scounts[size - 1] = N - (N / (size - 1)) * (size - 2);
    diff_time -= MPI_Wtime();
    do
    {
        if (rank == 0)
        {
            gen_time -= MPI_Wtime();
            for (int i = 0; i < N; ++i)
            {
                x[i] = -1 * drand48();
                y[i] = -1 * drand48();
                z[i] = -1 * drand48();
                //printf("%d (%f %f %f)\n", rank, x[i], y[i], z[i]);
            }
            gen_time += MPI_Wtime();
            //printf("\n\n");
        }
        MPI_Scatterv(x, scounts, displs, MPI_DOUBLE, x_sl, bufElems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(y, scounts, displs, MPI_DOUBLE, y_sl, bufElems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(z, scounts, displs, MPI_DOUBLE, z_sl, bufElems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank)
        {
            sum = 0;
            for (int i = 0; i < bufElems; ++i)
            {
                //printf("%d (%f %f %f)\n", rank, x_sl[i], y_sl[i], z_sl[i]);
                sum += x_sl[i] * x_sl[i] * x_sl[i] * y_sl[i] * y_sl[i] * z_sl[i];
            }
            //printf("\n\n");
        }
        MPI_Reduce(&sum, &glob_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (!rank)
        {
            total_N += N;
            acc_sum += glob_sum;
            flag = fabs(acc_sum / total_N - real_val) < eps;
        }
        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } while (!flag);
    diff_time += MPI_Wtime();
    MPI_Bcast(&gen_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    diff_time -= gen_time;
    //printf("%d time=%f\n", rank, diff_time);
    if (!rank)
    {
        diff_time = 0;
    }
    MPI_Reduce(&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank)
    {
        //printf("Gen_time = %f\n", gen_time);
        printf("Total_N=%d, value = %.10f,  error=%.10f, time=%f\n",
            total_N, acc_sum / total_N, fabs(acc_sum / total_N - real_val), max_time);
    }
    if (!rank)
    {
        free(x);
        free(y);
        free(z);
    }
    free(x_sl);
    free(y_sl);
    free(z_sl);
    free(displs);
    free(scounts);
    MPI_Finalize();
    return 0;
}
