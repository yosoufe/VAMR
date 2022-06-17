#pragma once
#include <Eigen/Dense>
#include "cuda_types.hpp"

Eigen::MatrixXd correlation(const Eigen::MatrixXd &input, const Eigen::MatrixXd &kernel);

Eigen::MatrixXd sobel_x_kernel();
Eigen::MatrixXd sobel_y_kernel();


#if WITH_CUDA
namespace cuda
{
    CuMatrixD sobel_x_kernel();
    CuMatrixD sobel_y_kernel();
    /**
     * @brief Cross correlation, with 0 padding at the input matrix
     * The output matrix would be the size of the input.
     * 
     * @param input 
     * @param kernel 
     * @return CuMatrixD 
     */
    CuMatrixD correlation(const CuMatrixD &input, const CuMatrixD &kernel);
    /**
     * @brief elemenet wise square in place
     */
    void ew_square(CuMatrixD &input);

    /**
     * @brief elementwise multiplication
     * 
     * @param i1 
     * @param i2 
     * @return cuda::CuMatrixD 
     */
    CuMatrixD ew_multiplication(const CuMatrixD &i1, const CuMatrixD &i2);

    CuMatrixD ones(int rows, int cols);
}


#endif