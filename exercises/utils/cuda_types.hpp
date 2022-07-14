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
        /**
         * @brief empty matrix with zero size.
         * To make sure the object has a correct deleter
         * 
         * @return CuMatrix 
         */
        CuMatrix();

        /**
         * @brief Construct a new Cu Matrix object
         * 
         * @param ptr 
         * @param n_cols 
         * @param n_rows 
         */
        CuMatrix(T *ptr, int n_cols, int n_rows);
        
        /**
         * @brief empty matrix with given rows and cols
         * 
         * @param cols 
         * @param rows 
         * @return CuMatrix 
         */
        CuMatrix(int cols, int rows);
        
        std::shared_ptr<T> d_data;
        int n_rows;
        int n_cols;
        void free();

        int n_elements() const
        {
            return n_rows * n_cols;
        }

        CuMatrix clone() const;
    };

    using CuMatrixD = CuMatrix<double>;
    using CuMatrixF = CuMatrix<float>;

    template <typename T>
    CuMatrix<T> eigen_to_cuda(const MatrixT<T> &eigen);

    template <typename T>
    MatrixT<T> cuda_to_eigen(const CuMatrix<T> &cuda_eigen);
}

bool are_matrices_close(const cuda::CuMatrixD &first, const Eigen::MatrixXd &second);

#endif