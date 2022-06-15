#include <gtest/gtest.h>
#include "cuda_types.hpp"
#include "operations.hpp"
#include "utils.hpp"
#include <Eigen/Dense>

#if WITH_CUDA

template <typename T1, typename T2>
bool are_matrices_close(const T1 &first, const T2 &second)
{
    return (first.cols() == second.cols() &&
            first.rows() == second.rows() &&
            (first - second).norm() < 1e-5);
}

TEST(UtilsTest, cuda_eigen)
{
    for (int i = 0; i < 1000; ++i)
    {
        Eigen::MatrixXd double_matrix = Eigen::MatrixXd::Random(5, 5);
        auto cuda_eigen_double = cuda::eigen_to_cuda(double_matrix);
        EXPECT_EQ(cuda_eigen_double.n_cols, double_matrix.cols());
        EXPECT_EQ(cuda_eigen_double.n_rows, double_matrix.rows());

        auto new_double_matrix = cuda::cuda_to_eigen(cuda_eigen_double);
        cuda_eigen_double.free();
        EXPECT_EQ(new_double_matrix.cols(), double_matrix.cols());
        EXPECT_EQ(new_double_matrix.rows(), double_matrix.rows());
        EXPECT_NEAR((new_double_matrix - double_matrix).norm(), 0.0, 1e-5);

        Eigen::MatrixXf float_matrix = Eigen::MatrixXf::Random(5, 5);
        auto cuda_eigen_float = cuda::eigen_to_cuda(float_matrix);
        EXPECT_EQ(cuda_eigen_float.n_cols, float_matrix.cols());
        EXPECT_EQ(cuda_eigen_float.n_rows, float_matrix.rows());

        auto new_float_matrix = cuda::cuda_to_eigen(cuda_eigen_float);
        cuda_eigen_float.free();
        EXPECT_TRUE(are_matrices_close(new_float_matrix, float_matrix));
    }
}

TEST(UtilsTest, cuda_sobel_kernel)
{
    Eigen::MatrixXd cpu_sobel_kernel_x = sobel_x_kernel();
    cuda::CuMatrixD cuda_sobel_kernel_x = cuda::sobel_x_kernel();
    auto gpu_copied_kernel = cuda::cuda_to_eigen(cuda_sobel_kernel_x);
    EXPECT_TRUE(are_matrices_close(gpu_copied_kernel, cpu_sobel_kernel_x));
    cuda_sobel_kernel_x.free();
}

TEST(UtilsTest, cuda_correlation)
{
    for (int i = 0; i < 10; ++i)
    {
        Eigen::MatrixXd kernel = sobel_x_kernel();
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(5, 5);
        auto correlated = correlation(matrix, kernel);
        cuda::CuMatrixD cuda_kernel = cuda::sobel_x_kernel();
        cuda::CuMatrixD cuda_matrix = cuda::eigen_to_cuda(matrix);
        cuda::CuMatrixD cuda_correlated = cuda::correlation(cuda_matrix, cuda_kernel);
        auto cuda_correlated_cpu = cuda::cuda_to_eigen(cuda_correlated);
        EXPECT_TRUE(are_matrices_close(
            cuda_correlated_cpu.block(1,1,3,3),
            correlated.block(1,1,3,3)));
        cuda_correlated.free();
        cuda_kernel.free();
        cuda_matrix.free();
        // std::cout << "cpu correlated:\n"
        //           << correlated << std::endl;
        // std::cout << "gpu correlated:\n"
        //           << cuda_correlated_cpu << std::endl;
    }
}

TEST(UtilsTest, cuda_ew_square)
{
    for (int i = 0; i< 100; ++i)
    {
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(500, 500);
        auto cpu_squared = matrix.array().square().matrix();
        cuda::CuMatrixD cuda_matrix = cuda::eigen_to_cuda(matrix);
        cuda::ew_square(cuda_matrix);
        auto gpu_squared = cuda::cuda_to_eigen(cuda_matrix);
        EXPECT_TRUE(are_matrices_close(gpu_squared, cpu_squared));
        // std::cout << "cpu squared\n" << cpu_squared << std::endl;
        // std::cout << "gpu squared\n" << gpu_squared << std::endl;
        cuda_matrix.free();
    }
}

#endif