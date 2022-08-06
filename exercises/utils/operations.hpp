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

    CuMatrixD ones(int rows, int cols);

    /**
     * @brief elementwise multiplication
     * 
     * @param i1 
     * @param i2 
     * @return CuMatrixD 
     */
    CuMatrixD operator* (const CuMatrixD &i1, const CuMatrixD &i2);
    CuMatrixD operator* (CuMatrixD &&i1, const CuMatrixD &i2);
    CuMatrixD operator* (const CuMatrixD &i1, CuMatrixD &&i2);
    CuMatrixD operator* (CuMatrixD &&i1, CuMatrixD &&i2);

    CuMatrixD operator+ (const CuMatrixD &i1, const CuMatrixD &i2);
    CuMatrixD operator+ (CuMatrixD &&i1, const CuMatrixD &i2);
    CuMatrixD operator+ (const CuMatrixD &i1, CuMatrixD &&i2);
    CuMatrixD operator+ (CuMatrixD &&i1, CuMatrixD &&i2);

    CuMatrixD operator- (const CuMatrixD &i1, const CuMatrixD &i2);
    CuMatrixD operator- (CuMatrixD &&i1, const CuMatrixD &i2);
    CuMatrixD operator- (const CuMatrixD &i1, CuMatrixD &&i2);
    CuMatrixD operator- (CuMatrixD &&i1, CuMatrixD &&i2);
    

    /**
     * @brief multiplication by scalar
     * 
     * @param y 
     * @param x 
     * @return CuMatrixD 
     */
    CuMatrixD operator* (const CuMatrixD& mat, double constant);
    CuMatrixD operator* (double constant, const CuMatrixD& mat);
    CuMatrixD operator* (CuMatrixD&& mat, double constant);
    CuMatrixD operator* (double constant, CuMatrixD&& mat);

    /**
     * @brief power function, in-place
     * 
     * @param i1 
     * @param pow 
     */
    CuMatrixD pow(CuMatrixD &&input, double pow);

    /**
     * @brief power function, out-of-place
     * 
     * @param i1 
     * @param pow 
     * @return CuMatrixD 
     */
    CuMatrixD pow (const CuMatrixD &i1, double pow);

    double norm(const CuMatrixD &input);

    CuMatrixD threshold_lower(const CuMatrixD &input, double threshold, double substitute);
    CuMatrixD threshold_lower(CuMatrixD &&input, double threshold, double substitute);

    void zero_borders(CuMatrixD &input, int s_row, int s_col, int l_row, int l_col);
}


#endif