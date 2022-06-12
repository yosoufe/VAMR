#pragma once 
#include <Eigen/Dense>

#if WITH_CUDA
namespace cuda
{
    template <typename T>
    using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    template <typename T>
    struct CuMatrix
    {
        T *d_data;
        int n_rows;
        int n_cols;

        void free();
    };

    template <typename T>
    CuMatrix<T> eigen_to_cuda(const MatrixT<T> &eigen);

    template <typename T>
    MatrixT<T> cuda_to_eigen(const CuMatrix<T> &cuda_eigen);

    using CuMatrixD = CuMatrix<double>;
    using CuMatrixF = CuMatrix<float>;
}

#endif