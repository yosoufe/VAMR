#include <gtest/gtest.h>
#include "utils.hpp"
#include <Eigen/Dense>

#if WITH_CUDA

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
        EXPECT_NEAR((new_double_matrix-double_matrix).norm(), 0.0, 1e-5);


        Eigen::MatrixXf float_matrix = Eigen::MatrixXf::Random(5, 5);
        auto cuda_eigen_float = cuda::eigen_to_cuda(float_matrix);
        EXPECT_EQ(cuda_eigen_float.n_cols, float_matrix.cols());
        EXPECT_EQ(cuda_eigen_float.n_rows, float_matrix.rows());

        auto new_float_matrix = cuda::cuda_to_eigen(cuda_eigen_float);
        cuda_eigen_float.free();
        EXPECT_EQ(new_float_matrix.cols(), float_matrix.cols());
        EXPECT_EQ(new_float_matrix.rows(), float_matrix.rows());
        EXPECT_NEAR((new_float_matrix-float_matrix).norm(), 0.0, 1e-5);
    }
}

#endif