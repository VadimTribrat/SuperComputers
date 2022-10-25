#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

int main(int argc, char * * argv)
{
    int rank, size, flag = 0, total_N = 0;
    const int N = 50;
    double eps = strtold(argv[1], NULL), glob_sum = 0, sum = 0, acc_sum = 0;
    double diff_time, max_time, start_time, end_time;
    const double real_val = 1.0 / 24;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    srand48(17);
    start_time = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    do
    {
        if (rank == 0)
        {
            double x[N * (size - 1)], y[N * (size - 1)], z[N * (size - 1)];
            int k = 1;
            sum = 0;
            for (int i = 0; i < N * (size - 1); ++i)
            {
                x[i] = -1 * drand48();
                y[i] = -1 * drand48();
                z[i] = -1 * drand48();
                if ((i + 1) ==  k * N)
                {
                    MPI_Send(x + (k - 1) * N, N, MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
                    MPI_Send(y + (k - 1) * N, N, MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
                    MPI_Send(z + (k - 1) * N, N, MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
                    k++;
                }
                //printf("%d (%f %f %f)\n", rank, x[i], y[i], z[i]);
                // if (k < size -1)
                // {
                //     if (i + 1 == k * N / (size - 1))
                //     {
                        // MPI_Send(x + (k - 1) * (N / (size - 1)), N / (size - 1), MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
                        // MPI_Send(y + (k - 1) * (N / (size - 1)), N / (size - 1), MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
                        // MPI_Send(z + (k - 1) * (N / (size - 1)), N / (size - 1), MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
                        // k++;
                //     }
                // }
                // else
                // {
                //     if (i == N - 1)
                //     {
                //         MPI_Send(x + (k - 1) * (N / (size - 1)), N - N / (size - 1) * (size - 2), MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
                //         MPI_Send(y + (k - 1) * (N / (size - 1)), N - N / (size - 1) * (size - 2), MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
                //         MPI_Send(z + (k - 1) * (N / (size - 1)), N - N / (size - 1) * (size - 2), MPI_DOUBLE, k, 0, MPI_COMM_WORLD); 
                //     }                   
                // }
            }
            //printf("\n\n");
        }
        else
        {
            MPI_Status status_x, status_y, status_z, status;
            int bufElems;
            sum = 0;
            MPI_Probe(0, 0, MPI_COMM_WORLD, &status );
            MPI_Get_count( &status, MPI_DOUBLE, &bufElems );
            double x[bufElems], y[bufElems], z[bufElems];
            MPI_Recv(x, bufElems, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status_x);
            MPI_Recv(y, bufElems, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status_y);
            MPI_Recv(z, bufElems, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status_z);
            for (int i = 0; i < bufElems; ++i)
            {
                //printf("%d (%f %f %f)\n", rank, x[i], y[i], z[i]);
                sum += x[i] * x[i] * x[i] * y[i] * y[i] * z[i];
            }
            //printf("\n\n");
        }
        total_N += N * (size - 1);
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
