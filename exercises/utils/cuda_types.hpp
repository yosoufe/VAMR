#pragma once
#include <Eigen/Dense>
#include "utils.hpp"

#if WITH_CUDA
namespace cuda
{
    template <typename T>
    struct CuMatrixDeleter
    {
        void operator()(T *p) const;
    };
    /**
     * @brief A struct to store the pointer to device memory
     * (on GPU) to the location of the Matrix
     * and dimensions of the matrix.
     *
     * @tparam T double or float.
     */
    template <typename T>
    struct CuMatrix
    {
    public:
        static CuMatrix factory(T *ptr, int n_cols, int n_rows);
        std::shared_ptr<T> d_data;
        int n_rows;
        int n_cols;
        void free();

    private:
        CuMatrix() = default;
    };

    using CuMatrixD = CuMatrix<double>;
    using CuMatrixF = CuMatrix<float>;

    template <typename T>
    CuMatrix<T> eigen_to_cuda(const MatrixT<T> &eigen);

    template <typename T>
    MatrixT<T> cuda_to_eigen(const CuMatrix<T> &cuda_eigen);
}

#endif