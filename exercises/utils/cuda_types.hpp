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
    class CuMatrix
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
         * @param rows
         * @param cols
         */
        CuMatrix(T *ptr, int rows, int cols);

        /**
         * @brief empty matrix with given rows and cols
         *
         * @param rows
         * @param cols
         * @return CuMatrix
         */
        CuMatrix(int rows, int cols);

        T *data() const { return d_data.get(); }
        int cols() const { return n_cols; }
        int rows() const { return n_rows; }

        int n_elements() const
        {
            return n_rows * n_cols;
        }

        CuMatrix clone() const;
        /**
         * @brief similar to Eigen::Matrix::block but
         * currently it always involves copying the data in GPU.
         *
         * @param row
         * @param col
         * @param height
         * @param width
         * @return CuMatrix
         */
        CuMatrix block(int row, int col, int height, int width) const;

        void free();
    private:
        std::shared_ptr<T> d_data;
        int n_rows;
        int n_cols;
        
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