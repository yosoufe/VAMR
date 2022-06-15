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
}


#endif