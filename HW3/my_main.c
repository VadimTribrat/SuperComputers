#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

const int N = 10, M = 10;
const double A = 4, B = 3;
double * x, *y;


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
void mul(double**, double);
double** sum_mat(double**, double**);
void delete_mat(double **);
void print_mat(double **);
double scalar_product(double **, double **);

int main()
{
    srand(42);
    double **w = create_mat(), **res, **residual;
    double **right_mat, **w_k, **op, **Ar, **true_res;
    //print_mat(w);
    true_res = create_mat();
    x = (double *) malloc((M + 1) * sizeof(double));
    y = (double *) malloc((N + 1) * sizeof(double));
    for (int i = 0; i <= M; ++i)
        x[i] = i * A / M;
    for (int i = 0; i <= N; ++i)
        y[i] = i * B / N;
    for (int i = 0; i <= M; ++i)
        for (int j = 0; j <= N; ++j)
            true_res[i][j] = true_func(x[i], y[j]);

    res = operator_A(w);
    right_mat = create_right_mat();
    printf("\n");
    mul(right_mat, -1);
    for (size_t i = 0; i < 10600; ++i)
    {
        op = operator_A(w);
        residual = sum_mat(op, right_mat);

        Ar = operator_A(residual);
        printf("%f\n", sqrt(scalar_product(residual, residual)));
        double coef = scalar_product(Ar, residual) / scalar_product(Ar, Ar);

        mul(residual, -coef);
        w_k = sum_mat(w, residual);
        delete_mat(w);
        delete_mat(op);
        delete_mat(residual);
        delete_mat(Ar);
        w = w_k;
    }
    // print_mat(w);
    mul(true_res, -1);
    op = sum_mat(w, true_res);
    printf("\n%.5f\n", sqrt(scalar_product(op, op)));
    // print_mat(true_res);

    free(x);
    free(y);
    delete_mat(w);
    delete_mat(op);
    delete_mat(res);
    delete_mat(right_mat);
    delete_mat(true_res);
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
    double ** mat = NULL;
    mat = (double * *) malloc((M + 1) * sizeof(double *));
    for (int i = 0; i <= M; ++i)
        mat[i] = (double *) malloc((N + 1) * sizeof(double));
    for (int i = 0; i <= M; ++i)
        for (int j = 0; j <= N; ++j)
            mat[i][j] = ((double) rand() / (RAND_MAX+1));
    return mat;
}

void delete_mat(double ** mat)
{
    for (int i = 0; i <= M; ++i)
        free(mat[i]);
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

double ** create_right_mat()
{
    double ** right_mat = create_mat();
    double h1 = x[1] - x[0], h2 = y[1] - y[0];
    for (int i = 1; i < M; ++i)
    {
        for (int j = 1; j < N; ++j)
        {
            right_mat[i][j] = F(x[i], y[j]);
        }
    }
    for (size_t i = 1; i < M; ++i)
    {
        right_mat[i][0] = F(x[i], y[0]) + 2 * phiB(x[i], y[0]) / h2;
        right_mat[i][N] = F(x[i], y[N]) + 2 * psiT(x[i], y[N]) / h2;
    }
    for (size_t j = 1; j < N; ++j)
    {
        right_mat[0][j] = F(x[0], y[j]) + 2 * phiL(x[0], y[j]) / h1;
        right_mat[M][j] = F(x[M], y[j]) + 2 * psiR(x[M], y[j]) / h1;
    }
    right_mat[0][0] = F(x[0], y[0]) + (2 / h1 + 2 / h2) * (h1 * phiB(x[0] + 1e-5, y[0]) + h2 * phiL(x[0], y[0] + 1e-5)) / (h1 + h2);
    right_mat[M][0] = F(x[M], y[0]) + (2 / h1 + 2 / h2) * (h1 * phiB(x[M] - 1e-5, y[0]) + h2 * psiR(x[M], y[0] + 1e-5)) / (h1 + h2);
    right_mat[0][N] = F(x[0], y[N]) + (2 / h1 + 2 / h2) * (h1 * psiT(x[0] + 1e-5, y[N]) + h2 * phiL(x[0], y[N] - 1e-5)) / (h1 + h2);
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