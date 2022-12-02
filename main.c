#include <iostream>
#include <ostream>
#include <cstddef>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <random>
#include <time.h>
#include <tuple>

using std::sqrt;

class Matrix
{
    double * * mat;
    size_t rowNum, colNum;
public:
    class Proxy
    {
        double * row;
        size_t colNum;
    public:
        Proxy(double * const, size_t);
        double& operator[](size_t);
        ~Proxy();
        Proxy(const Proxy&);
    };
    Matrix(int, int);
    ~Matrix();
    size_t getRows() const;
    size_t getColumns() const;
    Matrix& operator*=(double);
    static Matrix dot(Matrix&, Matrix&);
    static Matrix mat_mul(Matrix&, Matrix&);
    Matrix operator*(double);
    Matrix operator=(const Matrix&);
    bool operator==(const Matrix&);
    bool operator!=(const Matrix&);
    Matrix operator+(const Matrix&);
    Matrix operator-(const Matrix&);
    Matrix(const Matrix&);
    Proxy operator[](size_t) const;    
    friend std::ostream& operator<<(std::ostream&, const Matrix&);
};

double k(double, double);
double q(double, double);
double F(double, double);
double true_func(double, double);
double psi(double, double);
double scalar_product(Matrix&, Matrix&, double, double);
Matrix laplas_operator(Matrix&, double *, double *);
std::tuple<Matrix, Matrix> delta_w(Matrix&, double *, double *);
Matrix operatorA(Matrix&, double *, double *);
Matrix right(Matrix&, double *, double *);
Matrix getMatrixOperator(Matrix&, double *, double *);






int main()
{
    //srand(time(0));
    int gridX = 5, gridY = 5;
    double * x, * y, h1, h2;
    double A1 = 0, A2 = 4, B1 = 0, B2 = 3;
    Matrix w(gridX + 1, gridY + 1), r(gridX + 1, gridY + 1);
    Matrix true_res(gridX + 1, gridY + 1);
    Matrix w_k(gridX + 1, gridY + 1), w_k1(gridX + 1, gridY + 1);

    for (size_t i = 0; i < w.getRows(); ++i)
    {
        for (size_t j = 0; j < w.getColumns(); ++j)
        {
            w[i][j] = ((double) rand() / (RAND_MAX+1)) -1;
            //w[i][j] = 0; 
        }
    }
    h1 = (A2 - A1) / gridX;
    h2 = (B2 - B1) / gridY;
    x = (double *)malloc((gridX + 1) * sizeof(double));
    y = (double *)malloc((gridY + 1) * sizeof(double));
    for (int i = 0; i < gridX + 1; ++i)
        x[i] = A1 + i * h1;
    for (int i = 0; i < gridY + 1; ++i)
        y[i] = B1 + i * h2;
    for (size_t i = 0; i < w.getRows(); ++i)
    {
        for (size_t j = 0; j < w.getColumns(); ++j)
        {
            true_res[i][j] = true_func(x[i], y[j]);
        }
    }
    for (size_t i = 1; i < w.getRows() - 1; ++i)
    {
        true_res[i][0] = psi(x[i], y[0]);
    }
    //std::cout << "\n\n";
    auto [ax, by] = delta_w(w, x, y);

    Matrix right_part = right(w, x, y);
    auto lo = operatorA(w, x, y);
    std::cout << right_part.getRows() << " " << right_part.getColumns() << "\n";
    std::cout << lo.getRows() << " " << lo.getColumns() << "\n";
    double coef;
    w_k1 = w_k = w;
    // std::cout << w << "\n";
    // auto matA = getMatrixOperator(w, x, y);
    //std::cout << w <<"\n";
    // std::cout << operatorA(w, x, y) << "\n----------------------\n";
    // std::cout << Matrix::mat_mul(matA, w) << "\n\n";
    // std::cout << laplas_operator(w, x, y) << "\n";
    // std::cout << (ax + by) << "\n";
    for (size_t i = 0; i < 500; ++i)
    {
        Matrix residual = operatorA(w_k, x, y) - right_part;

        //std::cout << "Residual: " << std::sqrt(scalar_product(residual, residual, h1, h2)) << "\n";
        Matrix Ar = operatorA(residual, x, y);
        //std::cout << residual << "\n";
        coef = scalar_product(Ar, residual, h1, h2) / scalar_product(Ar, Ar, h1, h2);
        //coef = scalar_product(residual, residual, h1, h2) / scalar_product(Ar, residual, h1, h2);
        //std::cout << " coef: " << coef << "\n";
        residual = residual * (-coef);
        w_k1 = w_k + residual;
        std::cout << sqrt(scalar_product(w_k1, w_k, h1, h2)) << "\n";
        w_k = w_k1;
    }
    std::cout << operatorA(w_k, x, y) << "\n";
    std::cout << right_part << "\n";
    // std::cout << true_res << "\n";
    // std::cout << w_k << "\n";

    free(x);
    free(y);
    return 0;
}







double k(double x, double y)
{
    return 4 + x + y;
}

double q(double x, double y)
{
    return x + y;
}

double psi(double x, double y)
{
    return -x * (4 + x + y) / 4 + std::sqrt(4 + x*y);
}

double true_func(double x, double y)
{
    return std::sqrt(4 + x*y);
}

double F(double x, double y)
{
    double value = 0;
    value -= (x + y) / (2 * sqrt(4 + x*y));
    value += (4 + x + y) * (x *x + y * y) / (4 * sqrt((4 + x*y) * (4 + x*y) *(4 + x*y)));
    value += (x + y) * sqrt(4 + x*y);
    return value;
}

double scalar_product(Matrix& u, Matrix& v, double h1, double h2)
{
    int M = u.getRows() - 1, N = u.getColumns() -1 ;
    double rho1, rho2, value = 0;
    for (int i = 0; i <= M; ++i)
    {
        for (int j = 0; j <= N; ++j)
        {
            if (((i == 0) && (j > 0) && (j < N)) || ((i == M) && (j > 0) && (j < N)) || ((j == N) && (i > 0) && (i < M)))
                continue;
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

Matrix laplas_operator(Matrix& w, double * x, double * y)
{
    Matrix temp(w);
    double val1, val2;
    double h1 = x[1] - x[0], h2 = y[1] - y[0];
    for (size_t i = 1; i < w.getRows() - 1; ++i)
    {
        for (size_t j = 1; j < w.getColumns() - 1; ++j)
        {
            val1 = k(x[i] + 0.5 * h1, y[j]) * (w[i+1][j] - w[i][j]);
            val1 -= k(x[i] - 0.5 * h1, y[j]) * (w[i][j] - w[i-1][j]);
            val1 /= (h1 * h1);
            val2 = k(x[i], y[j] + 0.5 * h2) * (w[i][j+1] - w[i][j]);
            val2 -= k(x[i], y[j] - 0.5 * h2) * (w[i][j] - w[i][j-1]);
            val2 /= (h2 * h2);
            temp[i][j] = val1 + val2;
        }
    }
    return temp;
}

std::tuple<Matrix, Matrix> delta_w(Matrix& w,  double * x, double *y)
{
    double h1 = x[1] - x[0], h2 = y[1] - y[0];
    int gridX = w.getRows(), gridY = w.getColumns();
    Matrix temp_x(gridX, gridY), a(gridX, gridY),w_x(gridX, gridY),
     w_y(gridX, gridY), b(gridX, gridY),
    temp_y(gridX, gridY), res_x(gridX, gridY), res_y(gridX, gridY);
    for (size_t i = 1; i < w.getRows(); ++i)
    {
        for (size_t j = 0; j < w.getColumns(); ++j)
        {
            w_x[i][j] = (w[i][j] - w[i-1][j]) / h1;
            a[i][j] = k(x[i] - 0.5 * h1, y[j]);
        }
    } 
    for (size_t i = 0; i < w.getRows(); ++i)
    {
        for (size_t j = 1; j < w.getColumns(); ++j)
        {
            w_y[i][j] = (w[i][j] - w[i][j-1]) / h2;
            b[i][j] = k(x[i], y[j]- 0.5 * h2);
        }
    } 
    temp_x = Matrix::dot(a, w_x);
    temp_y = Matrix::dot(b, w_y);
    for (size_t i = 1; i < w.getRows() - 1; ++i)
    {
        double val1, val2;
        for (size_t j = 1; j < w.getColumns() - 1; ++j)
        {
            val1 = (temp_x[i+1][j] - temp_x[i][j]) / h1;
            val2 = (temp_y[i][j+1] - temp_y[i][j]) / h2;
            res_x[i][j] = val1;
            res_y[i][j] = val2;
        }
    } 
    return std::make_tuple(res_x, res_y);
}

Matrix operatorA(Matrix& w, double * x, double * y)
{

    for (size_t i = 0; i < w.getColumns(); ++i)
    {
        w[0][i] = true_func(x[0], y[i]);
        w[w.getRows()-1][i] = true_func(x[w.getRows() - 1], y[i]);
        w[i][w.getColumns()-1] = true_func(x[i], y[w.getColumns()-1]);
    }

    double h1 = x[1] - x[0], h2 = y[1] - y[0];
    int M = w.getRows()-1, N = w.getColumns()-1;
    auto [ax, by] = delta_w(w, x, y);
    Matrix mat1(M + 1, N + 1), mat2(M + 1, N + 1), F_mat(M + 1, N + 1), w_copy(w);
    std::vector<double> bottom(w.getRows(), 0);
    Matrix big_mat(M + 1, N + 1);
    Matrix res_mat(M - 1, N);
    double a1b1, a1b2, a2b1, a2b2;

    for (size_t i = 1; i < w_copy.getRows() - 1; ++i)
    {
        for (size_t j = 1; j < w_copy.getColumns() - 1; ++j)
        {
            w_copy[i][j] *= q(x[i], y[j]);
        }
    }
    mat1 = ax + by;
    mat1 *= -1;
    mat1 = mat1 + w_copy;
    //std::cout << mat1 << "\n";

    for (size_t i = 1; i < w_copy.getColumns() - 1; ++i)
    {
        bottom[i] = -2*by[i][1]/h2 + (q(x[i], y[0]) + 2/h2) * w[i][0] - ax[i][0]; 
    }
    
    // a1b1 = -2 * ax[1][0]/h1 - 2 * by[0][1]/h2 + (q(x[0], y[0]) + 2/h1 + 2/h2)*w[0][0];
    // a2b1 = 2 * ax[M][0]/h1 - 2 * by[M][1]/h2 + (q(x[M], y[0]) + 2/h1 + 2/h2)*w[M][0];
    // a2b2 = 2 * ax[M][N]/h1 + 2 * by[M][N]/h2 + (q(x[M], y[N]) + 2/h1 + 2/h2)*w[M][N];
    // a1b2 = -2 * ax[1][N]/h1 + 2 * by[0][N]/h2 + (q(x[0], y[N]) + 2/h1 + 2/h2)*w[0][N];

    for (size_t i = 1; i < w_copy.getRows()-1; ++i)
    {
        for (size_t j = 1; j < w_copy.getColumns()-1; ++j)
        {
            big_mat[i][j] = mat1[i][j];
        }
    }
    for (size_t i = 1; i < big_mat.getColumns() - 1; ++i)
    {
        big_mat[i][0] = bottom[i];
    }
    // big_mat[0][0] = a1b1;
    // big_mat[M][0] = a2b1;
    // big_mat[0][N] = a1b2;
    // big_mat[M][N] = a2b2;

    for (size_t i = 0; i < w.getColumns(); ++i)
    {
        big_mat[0][i] = true_func(x[0], y[i]);
        big_mat[w.getRows()-1][i] = true_func(x[w.getRows() - 1], y[i]);
        big_mat[i][w.getColumns()-1] = true_func(x[i], y[w.getColumns()-1]);
    }
    return big_mat; 
}

Matrix right(Matrix& w, double * x, double * y)
{
    int M = w.getRows()-1, N = w.getColumns()-1;
    Matrix right_mat(M + 1, N + 1);
    Matrix res_mat(M - 1, N);
    double h1 = x[1] - x[0], h2 = y[1] - y[0];
    for (size_t i = 1; i < w.getRows()-1; ++i)
    {
        for (size_t j = 1; j < w.getColumns()-1; ++j)
        {
            right_mat[i][j] = F(x[i], y[j]);
        }
    }
    for (size_t i = 1; i < w.getColumns()-1; ++i)
    {
        right_mat[i][0] = F(x[i], y[0]) + 2 * psi(x[i], y[0]) / h2;
    }
    right_mat[0][0] = F(x[0], y[0]) + (2/h1 + 2/h2) * psi(x[0], y[0]);
    right_mat[M][0] = F(x[M], y[0]) + (2/h1 + 2/h2) * psi(x[M], y[0]);
    right_mat[0][N] = F(x[0], y[N]) + (2/h1 + 2/h2) * psi(x[0], y[N]);
    right_mat[M][N] = F(x[M], y[N]) + (2/h1 + 2/h2) * psi(x[M], y[N]);
    for (size_t i = 0; i < w.getColumns(); ++i)
    {
        right_mat[0][i] = true_func(x[0], y[i]);
        right_mat[w.getRows()-1][i] = true_func(x[w.getRows() - 1], y[i]);
        right_mat[i][w.getColumns()-1] = true_func(x[i], y[w.getColumns()-1]);
    }
    return right_mat;
}

Matrix getMatrixOperator(Matrix& w, double *x, double *y)
{
    int M = w.getRows(), N = w.getColumns();
    Matrix temp(M, M), identity(M, N);
    Matrix res(M, N);
    for (size_t i = 0; i < w.getRows(); ++i)
    {
        identity[i][i] = 1;
    }
    auto val = operatorA(identity, x, y);
    auto example = operatorA(w, x, y);
    std::cout << val << "\n";
    // for (size_t i = 1; i < w.getRows()-1; ++i)
    //     for (size_t j = 1; j < w.getColumns()-1; ++j)
    //         res[i-1][j-1] = temp[i][j];
    std::cout << example << "\n\n" << Matrix::mat_mul(val, w) <<"\n";
    return val;
}












Matrix::Matrix(int row, int col)
{
    if (row <= 0 || col <= 0)
        throw std::exception();
    rowNum = static_cast<size_t>(row);
    colNum = static_cast<size_t>(col);
    mat = new double *[rowNum];
    for (size_t i = 0; i < rowNum; ++i)
    {
        mat[i] = new double[colNum];
        if (mat[i] == nullptr)
        {
            throw std::exception();
        }
    }
    for (size_t i = 0; i < rowNum; ++i)
        for (size_t j = 0; j < colNum; ++j)
            mat[i][j] = 0;
}

Matrix::~Matrix()
{
    for (size_t i = 0; i < rowNum; ++i)
        delete[] mat[i];
    delete[] mat;
}

size_t Matrix::getRows() const
{
    return rowNum;
}

size_t Matrix::getColumns() const
{
    return colNum;
}

Matrix& Matrix::operator*=(double mul)
{
    for (size_t i = 0; i < rowNum; ++i)
        for (size_t j = 0; j < colNum; ++j)
            mat[i][j] *= mul;
    return *this;   
}

bool Matrix::operator==(const Matrix& other)
{
    if (this->colNum != other.colNum || this->rowNum != other.rowNum)
        throw std::length_error("Inequal sizes");
    for (size_t i = 0; i < rowNum; ++i)
        for (size_t j = 0; j < colNum; ++j)
            if (mat[i][j] != other.mat[i][j])
                return false;
    return true;
}

bool Matrix::operator!=(const Matrix& other)
{
    if (this->colNum != other.colNum || this->rowNum != other.rowNum)
        throw std::length_error("Inequal sizes");
    return !(*this == other);
}

Matrix Matrix::operator+(const Matrix& other)
{
    if (this->colNum != other.colNum || this->rowNum != other.rowNum)
        throw std::length_error("Inequal sizes");
    Matrix temp(rowNum, colNum);
    for (size_t i = 0; i < rowNum; ++i)
        for (size_t j = 0; j < colNum; ++j)
            temp[i][j] = mat[i][j] + other.mat[i][j];
    return temp;
}

Matrix Matrix::operator-(const Matrix& other)
{
    if (this->colNum != other.colNum || this->rowNum != other.rowNum)
        throw std::length_error("Inequal sizes");
    Matrix temp(rowNum, colNum);
    for (size_t i = 0; i < rowNum; ++i)
        for (size_t j = 0; j < colNum; ++j)
            temp[i][j] = mat[i][j] - other.mat[i][j];
    return temp;
}

Matrix::Matrix(const Matrix& other)
{
    rowNum = other.rowNum;
    colNum = other.colNum;
    mat = new double *[rowNum];
    for (size_t i = 0; i < rowNum; ++i)
    {
        mat[i] = new double[colNum];
    }
    for (size_t i = 0; i < rowNum; ++i)
        for (size_t j = 0; j < colNum; ++j)
            mat[i][j] = other.mat[i][j];    
}
Matrix::Proxy Matrix::operator[](size_t i) const
{
    if (i >= rowNum)
        throw std::out_of_range("out og range");
    return Matrix::Proxy(mat[i], colNum);
}

Matrix::Proxy::Proxy(double * const ptr, size_t size)
{
    row = ptr;
    colNum = size;
}

Matrix::Proxy::Proxy(const Proxy& other)
{
    row = other.row;
    colNum = other.colNum;
}

Matrix::Proxy::~Proxy()
{
    row = nullptr;
}

double& Matrix::Proxy::operator[](size_t i)
{
    if (i >= colNum)
        throw std::out_of_range("out of range");
    return row[i];
}

std::ostream& operator<<(std::ostream& os, const Matrix& m)
{
    for (size_t i = 0; i < m.rowNum; ++i)
    {
        for (size_t j = 0; j < m.colNum; ++j)
            os << m.mat[i][j] << " ";
        os << "\n";
    }
    return os;
}

Matrix Matrix::dot(Matrix& mat1, Matrix& mat2)
{
    Matrix C(mat1.getRows(), mat2.getColumns());
    for (size_t i = 0; i < mat1.getRows(); ++i)
    {
        for (size_t j = 0; j < mat2.getColumns(); ++j)
        {
            C[i][j] = mat1[i][j] * mat2[i][j];
            // for (size_t k = 0; k < mat1.getColumns(); ++k)
            // {
            //     C[i][j] += mat1[i][k] * mat2[k][j];
            // }
        }
    }
    return C;
}

Matrix Matrix::mat_mul(Matrix& mat1, Matrix& mat2)
{
    Matrix C(mat1.getRows(), mat2.getColumns());
    for (size_t i = 0; i < mat1.getRows(); ++i)
    {
        for (size_t j = 0; j < mat2.getColumns(); ++j)
        {
            for (size_t k = 0; k < mat1.getColumns(); ++k)
            {
                C[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return C;
}

Matrix Matrix::operator*(double q)
{
    for (size_t i = 0; i < this->getRows(); ++i)
    {
        for (size_t j = 0; j < this->getColumns(); ++j)
        {
            mat[i][j] *= q;
        }
    }
    return *this;
}

Matrix Matrix::operator=(const Matrix& mat1)
{
    for (size_t i = 0; i < this->getRows(); ++i)
    {
        for (size_t j = 0; j < this->getColumns(); ++j)
        {
            mat[i][j] = mat1[i][j];
        }
    }
    return *this;   
}