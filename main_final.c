#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

int main(int argc, char** argv)
{
    int rank, size, flag = 0, total_N = 0;
    const int N = 10000;
    double eps = strtold(argv[1], NULL), glob_sum = 0, sum = 0, acc_sum = 0;
    double diff_time, max_time, start_time, end_time;
    const double real_val = 1.0 / 24;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    srand48(17);
    start_time = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double x[N], y[N], z[N];
    do
    {
        if (rank == 0)
        {
            sum = 0;
            for (int i = 0; i < N; ++i)
            {
                x[i] = -1 * drand48();
                y[i] = -1 * drand48();
                z[i] = -1 * drand48();
                //printf("%d (%f %f %f)\n", rank, x[i], y[i], z[i]);
            }
            for (size_t k = 1; k < size - 1; k++)
            {
                MPI_Send(x + (k - 1) * (N / (size - 1)), N / (size - 1), MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
                MPI_Send(y + (k - 1) * (N / (size - 1)), N / (size - 1), MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
                MPI_Send(z + (k - 1) * (N / (size - 1)), N / (size - 1), MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
            }
            //printf("\n\n");
            MPI_Send(x + (size - 2) * (N / (size - 1)), N - N / (size - 1) * (size - 2), MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD);
            MPI_Send(y + (size - 2) * (N / (size - 1)), N - N / (size - 1) * (size - 2), MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD);
            MPI_Send(z + (size - 2) * (N / (size - 1)), N - N / (size - 1) * (size - 2), MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD); 
        }
        else
        {
            MPI_Status status_x, status_y, status_z, status;
            int bufElems;
            sum = 0;
            MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &bufElems);
            double x_sl[bufElems], y_sl[bufElems], z_sl[bufElems];
            MPI_Recv(x_sl, bufElems, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status_x);
            MPI_Recv(y_sl, bufElems, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status_y);
            MPI_Recv(z_sl, bufElems, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status_z);
            for (int i = 0; i < bufElems; ++i)
            {
                //printf("%d (%f %f %f)\n", rank, x_sl[i], y_sl[i], z_sl[i]);
                sum += x_sl[i] * x_sl[i] * x_sl[i] * y_sl[i] * y_sl[i] * z_sl[i];
            }
            //printf("\n\n");
        }
        total_N += N;
        MPI_Allreduce(&sum, &glob_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        acc_sum += glob_sum;
        flag = fabs(acc_sum / total_N - real_val) < eps;
    } while (!flag);
    end_time = MPI_Wtime();
    diff_time = end_time - start_time;
    MPI_Reduce(&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank)
    {
        printf("Total_N=%d, value = %.10f,  error=%.10f, time=%f\n",
            total_N, acc_sum / total_N, fabs(acc_sum / total_N - real_val), max_time);
    }
    MPI_Finalize();
    return 0;
}
