#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h> 
#include <omp.h>

const int M = 50, N = 50;
const double A = 4, B = 3;
double * x, *y;
int ax1[2] = {0, 0}, ax2[2] = {0, 0};
double *right_r = NULL, *left_r = NULL, *top_r = NULL, *bottom_r = NULL;
omp_set_num_threads(4);


double k(double, double);
double F(double, double);
double q(double, double);
double true_func(double, double);
double psiR(double, double);
double psiT(double, double);
double phiL(double, double);
double phiB(double, double);
double right_x_deriv(double **w, int i, int j, double h);
double left_x_deriv(double **w, int i, int j, double h);
double right_y_deriv(double **w, int i, int j, double h);
double left_y_deriv(double **w, int i, int j, double h);
double a_x(double **, int, int, double);
double b_y(double **, int, int, double);
double ** operator_A(double **);
double ** create_right_mat();
double ** create_mat();
double ** create_rank_mat(double **);
void mul(double**, double);
double** sum_mat(double**, double**);
void delete_mat(double **);
void print_mat(double **);
double scalar_product(double **, double **);
double ** operator_A1(double ** w);

int main(int argc, char * * argv)
{
    int rank, size;
    MPI_Comm comm;
    MPI_Request requests[8];
    int dim[2] = {0, 0}, period[2] = {0, 0}, reorder = 0;
    int coord[2], id, top_scr, bttm_dest, right_dest, left_scr;

    srand(42);
    double **w, **res, **residual;
    double **right_mat, **w_k, **op, **Ar, **true_res;
    double **diff, val_diff;
    double time = 0, total_time;

    double *right_s = NULL, *left_s = NULL, *top_s = NULL, *bottom_s = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Dims_create(size, 2, dim);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &comm);

    MPI_Cart_coords(comm, rank, 2, coord);

    true_res = create_mat();
    w = create_mat();
    x = (double *) malloc((M + 1) * sizeof(double));
    y = (double *) malloc((N + 1) * sizeof(double));
    for (int i = 0; i <= M; ++i)
        x[i] = i * A / M;
    for (int i = 0; i <= N; ++i)
        y[i] = i * B / N;
    for (int i = 0; i <= M; ++i)
        for (int j = 0; j <= N; ++j)
            true_res[i][j] = true_func(x[i], y[j]);

    ax1[0] =  ((M + 1) / dim[0] + 1) * ((coord[0] < (M + 1) % dim[0]) ? coord[0] : ((M + 1) % dim[0])) + 
        (M + 1) / dim[0] * ((coord[0] >= (M + 1) % dim[0]) ? (abs(coord[0] - (M + 1) % dim[0])): 0);
    ax1[1] =  ax1[0] + (M + 1) / dim[0] + ((coord[0] < (M + 1) % dim[0]) ? 1 : 0);
    ax2[0] =  ((N + 1) / dim[1] + 1) * ((coord[1] < (N + 1) % dim[1]) ? coord[1] : ((N + 1) % dim[1])) + 
        (N + 1) / dim[1] * ((coord[1] >= (N + 1) % dim[1]) ? (abs(coord[1] - (N + 1) % dim[1])): 0);
    ax2[1] =  ax2[0] + (N + 1) / dim[1] + ((coord[1] < (N + 1) % dim[1]) ? 1 : 0);

    top_s = (double *) malloc((ax1[1] - ax1[0]) * sizeof(double));
    bottom_s = (double *) malloc((ax1[1] - ax1[0]) * sizeof(double));
    right_s = (double *) malloc((ax2[1] - ax2[0]) * sizeof(double));
    left_s = (double *) malloc((ax2[1] - ax2[0]) * sizeof(double));
    top_r = (double *) malloc((ax1[1] - ax1[0]) * sizeof(double));
    bottom_r = (double *) malloc((ax1[1] - ax1[0]) * sizeof(double));
    right_r = (double *) malloc((ax2[1] - ax2[0]) * sizeof(double));
    left_r = (double *) malloc((ax2[1] - ax2[0]) * sizeof(double));
    int counter = 0;
    do
    {
        time += -MPI_Wtime();
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Cart_coords(comm, rank, 2, coord);

        // printf("coord = {%d, %d}, ax1 = {%d, %d], ax2 = {%d, %d]\n", coord[0], coord[1], ax1[0], ax1[1], ax2[0], ax2[1]);

        MPI_Cart_shift(comm, 0, 1, &left_scr, &right_dest);
        MPI_Cart_shift(comm, 1, 1, &bttm_dest, &top_scr);

        if (top_scr != MPI_PROC_NULL)
        {
            for (int i = 0; i < ax1[1] - ax1[0]; ++i)
                top_s[i] = w[i + ax1[0]][ax2[1] - 1];
        }
        if (bttm_dest != MPI_PROC_NULL)
        {
            for (int i = 0; i < ax1[1] - ax1[0]; ++i)
            {
                bottom_s[i] = w[i + ax1[0]][ax2[0]];
            }
        }
        if (left_scr != MPI_PROC_NULL)
        {
            for (int i = 0; i < ax2[1] - ax2[0]; ++i)
                left_s[i] = w[ax1[0]][i + ax2[0]];
        }
        if (right_dest != MPI_PROC_NULL)
        {
            for (int i = 0; i < ax2[1] - ax2[0]; ++i)
                right_s[i] = w[ax1[1] - 1][i + ax2[0]];
        }

        MPI_Isend(top_s, (top_scr != MPI_PROC_NULL) ? ax1[1] - ax1[0] : 0, MPI_DOUBLE, top_scr, 0, comm, &requests[0]);
        MPI_Isend(bottom_s, (bttm_dest != MPI_PROC_NULL) ? ax1[1] - ax1[0] : 0, MPI_DOUBLE, bttm_dest, 0, comm, &requests[1]);
        MPI_Isend(left_s, (left_scr != MPI_PROC_NULL) ? ax2[1] - ax2[0] : 0, MPI_DOUBLE, left_scr, 0, comm, &requests[2]);
        MPI_Isend(right_s, (right_dest != MPI_PROC_NULL) ? ax2[1] - ax2[0] : 0, MPI_DOUBLE, right_dest, 0, comm, &requests[3]);

        MPI_Irecv(bottom_r, (bttm_dest != MPI_PROC_NULL) ? ax1[1] - ax1[0] : 0, MPI_DOUBLE, bttm_dest, 0, comm, &requests[4]);
        MPI_Irecv(top_r, (top_scr != MPI_PROC_NULL) ? ax1[1] - ax1[0] : 0, MPI_DOUBLE, top_scr, 0, comm, &requests[5]);
        MPI_Irecv(right_r, (right_dest != MPI_PROC_NULL) ? ax2[1] - ax2[0] : 0, MPI_DOUBLE, right_dest, 0, comm, &requests[6]);
        MPI_Irecv(left_r, (left_scr != MPI_PROC_NULL) ? ax2[1] - ax2[0] : 0, MPI_DOUBLE, left_scr, 0, comm, &requests[7]);

        int err = MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
        if (err == MPI_ERR_IN_STATUS) 
            printf("error\n");

        if (top_scr != MPI_PROC_NULL)
        {
            for (int i = 0; i < ax1[1] - ax1[0]; ++i)
                w[ax1[0] + i][ax2[1]] = top_r[i];
        }
        if (bttm_dest != MPI_PROC_NULL)
        {
            for (int i = 0; i < ax1[1] - ax1[0]; ++i)
                w[ax1[0] + i][ax2[0] - 1] = bottom_r[i];
        }
        if (left_scr != MPI_PROC_NULL)
        {
            for (int i = 0; i < ax2[1] - ax2[0]; ++i)
                w[ax1[0] - 1][ax2[0] + i] = left_r[i];
        }
        if (right_dest != MPI_PROC_NULL)
        {
            for (int i = 0; i < ax2[1] - ax2[0]; ++i)
                w[ax1[1]][ax2[0] + i] = right_r[i];
        }  

        res = operator_A(w);
        right_mat = create_right_mat();
        // printf("coord = {%d, %d}, ax1 = {%d, %d], ax2 = {%d, %d]\n", coord[0], coord[1], ax1[0], ax1[1], ax2[0], ax2[1]);
        // print_mat(right_mat);
        // printf("\n");
        time += MPI_Wtime();
        if (rank != 0)
        {
            MPI_Send(&res[0][0], (M + 1) * (N + 1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&right_mat[0][0], (M + 1) * (N + 1), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            double **big_mat = create_mat(), **big_right = create_mat(), **temp_mat = create_mat();
            for (int k = ax1[0]; k < ax1[1]; ++k)
                for (int j = ax2[0]; j < ax2[1]; ++j)
                {
                    big_mat[k][j] = res[k][j];
                    big_right[k][j] = right_mat[k][j];
                }
            for (int i = 1; i < size; ++i)
            {
                MPI_Cart_coords(comm, i, 2, coord);
                MPI_Recv(&temp_mat[0][0], (M + 1) * (N + 1), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                ax1[0] =  ((M + 1) / dim[0] + 1) * ((coord[0] < (M + 1) % dim[0]) ? coord[0] : ((M + 1) % dim[0])) + 
                    (M + 1) / dim[0] * ((coord[0] >= (M + 1) % dim[0]) ? (abs(coord[0] - (M + 1) % dim[0])): 0);
                ax1[1] =  ax1[0] + (M + 1) / dim[0] + ((coord[0] < (M + 1) % dim[0]) ? 1 : 0);
                ax2[0] =  ((N + 1) / dim[1] + 1) * ((coord[1] < (N + 1) % dim[1]) ? coord[1] : ((N + 1) % dim[1])) + 
                    (N + 1) / dim[1] * ((coord[1] >= (N + 1) % dim[1]) ? (abs(coord[1] - (N + 1) % dim[1])): 0);
                ax2[1] =  ax2[0] + (N + 1) / dim[1] + ((coord[1] < (N + 1) % dim[1]) ? 1 : 0);
                for (int k = ax1[0]; k < ax1[1]; ++k)
                    for (int j = ax2[0]; j < ax2[1]; ++j)
                        big_mat[k][j] = temp_mat[k][j];
                MPI_Recv(&temp_mat[0][0], (M + 1) * (N + 1), MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int k = ax1[0]; k < ax1[1]; ++k)
                    for (int j = ax2[0]; j < ax2[1]; ++j)
                        big_right[k][j] = temp_mat[k][j];

                MPI_Cart_coords(comm, 0, 2, coord);
                ax1[0] =  ((M + 1) / dim[0] + 1) * ((coord[0] < (M + 1) % dim[0]) ? coord[0] : ((M + 1) % dim[0])) + 
                    (M + 1) / dim[0] * ((coord[0] >= (M + 1) % dim[0]) ? (abs(coord[0] - (M + 1) % dim[0])): 0);
                ax1[1] =  ax1[0] + (M + 1) / dim[0] + ((coord[0] < (M + 1) % dim[0]) ? 1 : 0);
                ax2[0] =  ((N + 1) / dim[1] + 1) * ((coord[1] < (N + 1) % dim[1]) ? coord[1] : ((N + 1) % dim[1])) + 
                    (N + 1) / dim[1] * ((coord[1] >= (N + 1) % dim[1]) ? (abs(coord[1] - (N + 1) % dim[1])): 0);
                ax2[1] =  ax2[0] + (N + 1) / dim[1] + ((coord[1] < (N + 1) % dim[1]) ? 1 : 0);

            }
            // ax1[0] = 0; ax1[1] = M + 1;
            // ax2[0] = 0; ax2[1] = N + 1;
            //print_mat(big_mat);
            // printf("\n");
            mul(big_right, -1);
            //print_mat(big_right);
            residual = sum_mat(big_mat, big_right);

            // printf("\n");
            // print_mat(residual);

            Ar = operator_A1(residual);
            printf("%f\n", sqrt(scalar_product(residual, residual)));
            double coef = scalar_product(Ar, residual) / scalar_product(Ar, Ar);

            mul(residual, -coef);
            w_k = sum_mat(w, residual);
            mul(w_k, -1);
            diff = sum_mat(w, w_k);
            val_diff = sqrt(scalar_product(diff, diff));
            delete_mat(diff);
            mul(w_k, -1);
            delete_mat(w);
            //delete_mat(op);
            delete_mat(residual);
            delete_mat(Ar);
            w = w_k;
            // print_mat(big_mat);
            // printf("\n");
            // print_mat(big_right);
            delete_mat(temp_mat); 
            delete_mat(big_mat);
            delete_mat(big_right);     
            //print_mat(w);
            //printf("\n"); 
        }
        MPI_Bcast(&val_diff, 1, MPI_DOUBLE, 0, comm);
        MPI_Bcast(&w[0][0], (M + 1) * (N + 1), MPI_DOUBLE, 0, comm);
    }
    while(val_diff >= 1e-5);
    MPI_Reduce(&time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank)
    {
        mul(true_res, -1);
        diff = sum_mat(w, true_res);
        val_diff = sqrt(scalar_product(diff, diff));
        printf("Time = %.2f, diff = %e\n", total_time, val_diff);
        delete_mat(diff);
    }

    free(x);
    free(y);
    free(top_s);
    free(bottom_s);
    free(right_s);
    free(left_s);
    free(top_r);
    free(bottom_r);
    free(right_r);
    free(left_r);
    delete_mat(w);
    //delete_mat(op);
    delete_mat(res);
    delete_mat(right_mat);
    delete_mat(true_res);
    MPI_Finalize();
    return 0;
}

double k(double x, double y)
{
    return 4 + x + y;
}

double psiR(double x, double y)
{
    return (8 + y) * y / (4 * sqrt(1 + y)) + 2 * sqrt(1 + y);
}
double psiT(double x, double y)
{
    return (7 + x) * x / (2 * sqrt(4 + 3 * x));
}

double phiL(double x, double y)
{
    return -(4 + y) * y / 4 + 2;
}

double phiB(double x, double y)
{
    return -(4 + x) * x / 4;
}

double q(double x, double y)
{
    return x + y;
}

double true_func(double x, double y)
{
    return sqrt(4 + x * y);
}

double F(double x, double y)
{
    double value = 0;
    value += (4 + x + y) * (x * x + y * y) / (4 * sqrt((4 + x * y) * (4 + x * y) * (4 + x * y)));
    value -= (x + y) / (2 * sqrt(4 + x * y));
    value += (x + y) * sqrt(4 + x * y);
    return value;
}

double scalar_product(double ** u, double ** v)
{
    double rho1, rho2, value = 0;
    double h1 = x[1] - x[0], h2 = y[1] - y[0];
    for (int i = 0; i <= M; ++i)
    {
        for (int j = 0; j <= N; ++j)
        {
            if ((i <= M - 1) && (i >= 1))
                rho1 = 1;
            else
                rho1 = 0.5;
            if ((j <= N - 1) && (j >= 1))
                rho2 = 1;
            else
                rho2 = 0.5;
            value += h1 * h2 * rho1 * rho2 * u[i][j] * v[i][j];
        }
    }
    return value;
}


double ** create_mat()
{
    double ** mat = NULL, *data, val = 0 ;
    data = (double *)malloc((M + 1) * (N + 1) * sizeof(double));
    mat = (double * *) malloc((M + 1) * sizeof(double *));
    for (int i = 0; i <= M; ++i)
        mat[i] = &data[(N + 1) * i];
    for (int i = 0; i <= M; ++i)
        for (int j = 0; j <= N; ++j)
            mat[i][j] = val++;
            //mat[i][j] = ((double) rand() / (RAND_MAX+1));
    return mat;
}

void delete_mat(double ** mat)
{
    free(mat[0]);
    free(mat);
}

void print_mat(double ** mat)
{
    for (int i = 0; i <= M; ++i)
    {
        for (int j = 0; j <= N; ++j)
            printf("%.5f ", mat[i][j]);
        printf("\n");
    }
    // for (int i = ax1[0]; i < ax1[1]; ++i)
    // {
    //     for (int j = ax2[0]; j < ax2[1]; ++j)
    //         printf("%.5f ", mat[i][j]);
    //     printf("\n");
    // }
}

double right_x_deriv(double **w, int i, int j, double h)
{
    return (w[i + 1][j] - w[i][j]) / h;
}

double right_y_deriv(double **w, int i, int j, double h)
{
    return (w[i][j + 1] - w[i][j]) / h;
}

double left_x_deriv(double **w, int i, int j, double h)
{
    return (w[i][j] - w[i - 1][j]) / h;
}

double left_y_deriv(double **w, int i, int j, double h)
{
    return (w[i][j] - w[i][j - 1]) / h;
}

double a_x(double **w, int i, int j, double h1)
{
    return 1/h1 * (k(x[i] + 0.5*h1, y[j]) * left_x_deriv(w, i + 1, j, h1) - k(x[i] - 0.5*h1, y[j]) * left_x_deriv(w, i, j, h1));
}

double b_y(double **w, int i, int j, double h2)
{
    return 1/h2 * (k(x[i], y[j] + 0.5*h2) * left_y_deriv(w, i, j + 1, h2) - k(x[i], y[j] - 0.5*h2) * left_y_deriv(w, i, j, h2));
}

double ** operator_A(double ** w)
{
    double ** temp_mat = create_mat();
    double h1 = x[1] - x[0], h2 = y[1] - y[0];
    double a1b1, a1b2, a2b1, a2b2;
    if ((ax1[0] == 0) && ((ax2[1] - 1) == N))
    {
        #pragma omp parallel for
        for (int i = ax1[0] + 1; i < ax1[1]; ++i)
        {
            #pragma omp parallel for
            for (int j = ax2[0]; j < ax2[1] - 1; ++j)
            {
                temp_mat[i][j] = -(a_x(w, i, j, h1) + b_y(w, i, j, h2)) + w[i][j] * q(x[i], y[j]);
            }
        }
    }
    else if ((ax1[0] == 0) && (ax2[0] == 0))
    {
        #pragma omp parallel for
        for (int i = ax1[0] + 1; i < ax1[1]; ++i)
        {
            #pragma omp parallel for
            for (int j = ax2[0] + 1; j < ax2[1]; ++j)
            {
                temp_mat[i][j] = -(a_x(w, i, j, h1) + b_y(w, i, j, h2)) + w[i][j] * q(x[i], y[j]);
            }
        }
    }
    else if (((ax1[1] - 1) == M) && (ax2[0] == 0))
    {
        #pragma omp parallel for
        for (int i = ax1[0]; i < ax1[1] - 1; ++i)
        {
            #pragma omp parallel for
            for (int j = ax2[0] + 1; j < ax2[1]; ++j)
            {
                temp_mat[i][j] = -(a_x(w, i, j, h1) + b_y(w, i, j, h2)) + w[i][j] * q(x[i], y[j]);
            }
        }
    }
    else if (((ax1[1] - 1) == M) && ((ax2[1] - 1) == N))
    {
        #pragma omp parallel for
        for (int i = ax1[0]; i < ax1[1] - 1; ++i)
        {
            #pragma omp parallel for
            for (int j = ax2[0]; j < ax2[1] - 1; ++j)
            {
                temp_mat[i][j] = -(a_x(w, i, j, h1) + b_y(w, i, j, h2)) + w[i][j] * q(x[i], y[j]);
            }
        }
    }
    else if (((ax1[1] - 1) == M))
    {
        #pragma omp parallel for
        for (int i = ax1[0]; i < ax1[1] - 1; ++i)
        {
            #pragma omp parallel for
            for (int j = ax2[0]; j < ax2[1]; ++j)
            {
                temp_mat[i][j] = -(a_x(w, i, j, h1) + b_y(w, i, j, h2)) + w[i][j] * q(x[i], y[j]);
            }
        }
    }
    else if ((ax1[0] == 0))
    {
        #pragma omp parallel for
        for (int i = ax1[0] + 1; i < ax1[1]; ++i)
        {
            #pragma omp parallel for
            for (int j = ax2[0]; j < ax2[1]; ++j)
            {
                temp_mat[i][j] = -(a_x(w, i, j, h1) + b_y(w, i, j, h2)) + w[i][j] * q(x[i], y[j]);
            }
        }
    }
    else if ((ax2[0] == 0))
    {
        #pragma omp parallel for
        for (int i = ax1[0]; i < ax1[1]; ++i)
        {
            #pragma omp parallel for
            for (int j = ax2[0] + 1; j < ax2[1]; ++j)
            {
                temp_mat[i][j] = -(a_x(w, i, j, h1) + b_y(w, i, j, h2)) + w[i][j] * q(x[i], y[j]);
            }
        }
    }
    else if (((ax2[1] - 1) == N))
    {
        #pragma omp parallel for
        for (int i = ax1[0]; i < ax1[1]; ++i)
        {
            #pragma omp parallel for
            for (int j = ax2[0]; j < ax2[1] - 1; ++j)
            {
                temp_mat[i][j] = -(a_x(w, i, j, h1) + b_y(w, i, j, h2)) + w[i][j] * q(x[i], y[j]);
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (int i = ax1[0]; i < ax1[1]; ++i)
        {
            #pragma omp parallel for
            for (int j = ax2[0]; j < ax2[1]; ++j)
            {
                temp_mat[i][j] = -(a_x(w, i, j, h1) + b_y(w, i, j, h2)) + w[i][j] * q(x[i], y[j]);
            }
        }
    }
    if (ax2[0] == 0)
        for (int i = 1; i < M; ++i)
            temp_mat[i][0] = -2 * k(x[i], y[1] - h2 * 0.5) * left_y_deriv(w, i, 1, h2) / h2 + q(x[i], y[0]) * w[i][0] - a_x(w, i, 0, h1); 
    if ((ax2[1] - 1) == N)
        for (int i = 1; i < M; ++i)
            temp_mat[i][N] = 2 * k(x[i], y[N] - h2 * 0.5) * left_y_deriv(w, i, N, h2) / h2 + q(x[i], y[N]) * w[i][N] - a_x(w, i, N, h1);
    if (ax1[0] == 0)
        for (size_t j = 1; j < N; ++j)
            temp_mat[0][j] = -2 * k(x[1] - h1 * 0.5, y[j]) * left_x_deriv(w, 1, j, h1) / h1 + (q(x[0], y[j]) + 2 / h1) * w[0][j] - b_y(w, 0, j, h2);
    if ((ax1[1] - 1) == M)
    for (size_t j = 1; j < N; ++j)
        temp_mat[M][j] = 2 * k(x[M] - h1 * 0.5, y[j]) * left_x_deriv(w, M, j, h1) / h1 + (q(x[M], y[j]) + 2 / h1) * w[M][j] - b_y(w, M, j, h2);
    if ((ax1[0] == 0) && (ax2[0] == 0))
        temp_mat[0][0] = -2 * k(x[1] - h1 * 0.5, y[0]) * left_x_deriv(w, 1, 0, h1) / h1 - 2 * k(x[0], y[1] - h2 * 0.5) * left_y_deriv(w, 0, 1, h2) / h2 + (q(x[0], y[0]) + 2 / h1 + 2 / h2 * 0) * w[0][0];
    if (((ax1[1] - 1) == M) && (ax2[0] == 0))
        temp_mat[M][0] = 2 * k(x[M] - h1 * 0.5, y[0]) * left_x_deriv(w, M, 0, h1) / h1 - 2 * k(x[M], y[1] - h2 * 0.5) * left_y_deriv(w, M, 1, h2) / h2 + (q(x[M], y[0]) + 2 / h1 + 2 / h2 * 0) * w[M][0];
    if (((ax1[1] - 1) == M) && ((ax2[1] - 1) == N))
        temp_mat[M][N] = 2 * k(x[M] - h1 * 0.5, y[N]) * left_x_deriv(w, M, N, h1) / h1 + 2 * k(x[M], y[N] - h2 * 0.5) * left_y_deriv(w, M, N, h2) / h2 + (q(x[M], y[N]) + 2 / h1 + 2 / h2 * 0) * w[M][N];
    if ((ax1[0] == 0) && ((ax2[1] - 1) == N))
        temp_mat[0][N] = -2 * k(x[1] - h1 * 0.5, y[N]) * left_x_deriv(w, 1, N, h1) / h1 + 2 * k(x[0], y[N] - h2 * 0.5) * left_y_deriv(w, 0, N, h2) / h2 + (q(x[0], y[N]) + 2 / h1 + 2 / h2 * 0) * w[0][N];
    return temp_mat;
}

double ** create_right_mat()
{
    double ** right_mat = create_mat();
    double h1 = x[1] - x[0], h2 = y[1] - y[0];

    if ((ax1[0] == 0) && ((ax2[1] - 1) == N))
        for (int i = ax1[0] + 1; i < ax1[1]; ++i)
        {
            for (int j = ax2[0]; j < ax2[1] - 1; ++j)
            {
                right_mat[i][j] = F(x[i], y[j]);
            }
        }
    else if ((ax1[0] == 0) && (ax2[0] == 0))
        for (int i = ax1[0] + 1; i < ax1[1]; ++i)
        {
            for (int j = ax2[0] + 1; j < ax2[1]; ++j)
            {
                right_mat[i][j] = F(x[i], y[j]);
            }
        }
    else if (((ax1[1] - 1) == M) && (ax2[0] == 0))
        for (int i = ax1[0]; i < ax1[1] - 1; ++i)
        {
            for (int j = ax2[0] + 1; j < ax2[1]; ++j)
            {
                right_mat[i][j] = F(x[i], y[j]);
            }
        }
    else if (((ax1[1] - 1) == M) && ((ax2[1] - 1) == N))
        for (int i = ax1[0]; i < ax1[1] - 1; ++i)
        {
            for (int j = ax2[0]; j < ax2[1] - 1; ++j)
            {
                right_mat[i][j] = F(x[i], y[j]);
            }
        }
    else if (((ax1[1] - 1) == M))
        for (int i = ax1[0]; i < ax1[1] - 1; ++i)
        {
            for (int j = ax2[0]; j < ax2[1]; ++j)
            {
                right_mat[i][j] = F(x[i], y[j]);
            }
        }
    else if ((ax1[0] == 0))
        for (int i = ax1[0] + 1; i < ax1[1]; ++i)
        {
            for (int j = ax2[0]; j < ax2[1]; ++j)
            {
                right_mat[i][j] = F(x[i], y[j]);
            }
        }
    else if ((ax2[0] == 0))
        for (int i = ax1[0]; i < ax1[1]; ++i)
        {
            for (int j = ax2[0] + 1; j < ax2[1]; ++j)
            {
                right_mat[i][j] = F(x[i], y[j]);
            }
        }
    else if (((ax2[1] - 1) == N))
        for (int i = ax1[0]; i < ax1[1]; ++i)
        {
            for (int j = ax2[0]; j < ax2[1] - 1; ++j)
            {
                right_mat[i][j] = F(x[i], y[j]);
            }
        }
    else
        for (int i = ax1[0]; i < ax1[1]; ++i)
        {
            for (int j = ax2[0]; j < ax2[1]; ++j)
            {
                right_mat[i][j] = F(x[i], y[j]);
            }
        }
    if (ax2[0] == 0)
        for (size_t i = 1; i < M; ++i)
            right_mat[i][0] = F(x[i], y[0]) + 2 * phiB(x[i], y[0]) / h2;
    if ((ax2[1] - 1) == N)
        for (size_t i = 1; i < M; ++i)
            right_mat[i][N] = F(x[i], y[N]) + 2 * psiT(x[i], y[N]) / h2;
    if (ax1[0] == 0)
        for (size_t j = 1; j < N; ++j)
            right_mat[0][j] = F(x[0], y[j]) + 2 * phiL(x[0], y[j]) / h1;
    if ((ax1[1] - 1) == M)
        for (size_t j = 1; j < N; ++j)
            right_mat[M][j] = F(x[M], y[j]) + 2 * psiR(x[M], y[j]) / h1;
    if ((ax1[0] == 0) && (ax2[0] == 0))
        right_mat[0][0] = F(x[0], y[0]) + (2 / h1 + 2 / h2) * (h1 * phiB(x[0] + 1e-5, y[0]) + h2 * phiL(x[0], y[0] + 1e-5)) / (h1 + h2);
    if (((ax1[1] - 1) == M) && (ax2[0] == 0))
        right_mat[M][0] = F(x[M], y[0]) + (2 / h1 + 2 / h2) * (h1 * phiB(x[M] - 1e-5, y[0]) + h2 * psiR(x[M], y[0] + 1e-5)) / (h1 + h2);
    if ((ax1[0] == 0) && ((ax2[1] - 1) == N))
        right_mat[0][N] = F(x[0], y[N]) + (2 / h1 + 2 / h2) * (h1 * psiT(x[0] + 1e-5, y[N]) + h2 * phiL(x[0], y[N] - 1e-5)) / (h1 + h2);
    if (((ax1[1] - 1) == M) && ((ax2[1] - 1) == N))
        right_mat[M][N] = F(x[M], y[N]) + (2 / h1 + 2 / h2) * (h1 * psiT(x[M] - 1e-5, y[N]) + h2 * psiR(x[M], y[N] - 1e-5)) / (h1 + h2);
    return right_mat;
}

void mul(double **mat, double coef)
{
    for (int i = 0; i <= M; ++i)
        for (int j = 0; j <= N; ++j)
            mat[i][j] = mat[i][j] * coef;
}
double** sum_mat(double **mat1, double **mat2)
{
    double ** temp = create_mat();
    for (int i = 0; i <= M; ++i)
        for (int j = 0; j <= N; ++j)
            temp[i][j] = mat1[i][j] + mat2[i][j];
    return temp;
}

double ** operator_A1(double ** w)
{
    double ** temp_mat = create_mat();
    double h1 = x[1] - x[0], h2 = y[1] - y[0];
    double a1b1, a1b2, a2b1, a2b2;
    for (int i = 1; i < M; ++i)
    {
        for (int j = 1; j < N; ++j)
        {
            temp_mat[i][j] = -(a_x(w, i, j, h1) + b_y(w, i, j, h2)) + w[i][j] * q(x[i], y[j]);
        }
    }
    for (int i = 1; i < M; ++i)
    {
        temp_mat[i][0] = -2 * k(x[i], y[1] - h2 * 0.5) * left_y_deriv(w, i, 1, h2) / h2 + q(x[i], y[0]) * w[i][0] - a_x(w, i, 0, h1); 
        temp_mat[i][N] = 2 * k(x[i], y[N] - h2 * 0.5) * left_y_deriv(w, i, N, h2) / h2 + q(x[i], y[N]) * w[i][N] - a_x(w, i, N, h1);
    }

    for (size_t j = 1; j < N; ++j)
    {
        temp_mat[0][j] = -2 * k(x[1] - h1 * 0.5, y[j]) * left_x_deriv(w, 1, j, h1) / h1 + (q(x[0], y[j]) + 2 / h1) * w[0][j] - b_y(w, 0, j, h2);
        temp_mat[M][j] = 2 * k(x[M] - h1 * 0.5, y[j]) * left_x_deriv(w, M, j, h1) / h1 + (q(x[M], y[j]) + 2 / h1) * w[M][j] - b_y(w, M, j, h2);
    }

    temp_mat[0][0] = -2 * k(x[1] - h1 * 0.5, y[0]) * left_x_deriv(w, 1, 0, h1) / h1 - 2 * k(x[0], y[1] - h2 * 0.5) * left_y_deriv(w, 0, 1, h2) / h2 + (q(x[0], y[0]) + 2 / h1 + 2 / h2 * 0) * w[0][0];
    temp_mat[M][0] = 2 * k(x[M] - h1 * 0.5, y[0]) * left_x_deriv(w, M, 0, h1) / h1 - 2 * k(x[M], y[1] - h2 * 0.5) * left_y_deriv(w, M, 1, h2) / h2 + (q(x[M], y[0]) + 2 / h1 + 2 / h2 * 0) * w[M][0];
    temp_mat[M][N] = 2 * k(x[M] - h1 * 0.5, y[N]) * left_x_deriv(w, M, N, h1) / h1 + 2 * k(x[M], y[N] - h2 * 0.5) * left_y_deriv(w, M, N, h2) / h2 + (q(x[M], y[N]) + 2 / h1 + 2 / h2 * 0) * w[M][N];
    temp_mat[0][N] = -2 * k(x[1] - h1 * 0.5, y[N]) * left_x_deriv(w, 1, N, h1) / h1 + 2 * k(x[0], y[N] - h2 * 0.5) * left_y_deriv(w, 0, N, h2) / h2 + (q(x[0], y[N]) + 2 / h1 + 2 / h2 * 0) * w[0][N];
    return temp_mat;
}
