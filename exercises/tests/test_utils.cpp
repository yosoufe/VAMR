#include <gtest/gtest.h>
#include "cuda_types.hpp"
#include "operations.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include "folder_manager.hpp"

#if WITH_CUDA

TEST(UtilsTest, cuda_eigen)
{
    for (int i = 0; i < 10; ++i)
    {
        Eigen::MatrixXd double_matrix = Eigen::MatrixXd::Random(5, 5);
        auto cuda_eigen_double = cuda::eigen_to_cuda(double_matrix);
        EXPECT_EQ(cuda_eigen_double.n_cols, double_matrix.cols());
        EXPECT_EQ(cuda_eigen_double.n_rows, double_matrix.rows());

        auto new_double_matrix = cuda::cuda_to_eigen(cuda_eigen_double);
        EXPECT_EQ(new_double_matrix.cols(), double_matrix.cols());
        EXPECT_EQ(new_double_matrix.rows(), double_matrix.rows());
        EXPECT_NEAR((new_double_matrix - double_matrix).norm(), 0.0, 1e-5);

        Eigen::MatrixXf float_matrix = Eigen::MatrixXf::Random(5, 5);
        auto cuda_eigen_float = cuda::eigen_to_cuda(float_matrix);
        EXPECT_EQ(cuda_eigen_float.n_cols, float_matrix.cols());
        EXPECT_EQ(cuda_eigen_float.n_rows, float_matrix.rows());

        auto new_float_matrix = cuda::cuda_to_eigen(cuda_eigen_float);
        EXPECT_TRUE(are_matrices_close(new_float_matrix, float_matrix));
    }
}

TEST(UtilsTest, cuda_img)
{
    std::string in_data_root = "../../data/ex03/";
    SortedImageFiles image_files(in_data_root);
    auto src_img = cv::imread(image_files[0].path(), cv::IMREAD_GRAYSCALE);
    Eigen::MatrixXd eigen_img = cv_2_eigen(src_img);
    // std::cout << eigen_img.Flags << std::endl;
    // std::cout << (eigen_img.Flags & Eigen::RowMajorBit) << std::endl;
    auto d_eigen_img = cuda::eigen_to_cuda(eigen_img);
    auto hd_eigen_img = cuda::cuda_to_eigen(d_eigen_img);
    EXPECT_TRUE(are_matrices_close(hd_eigen_img, eigen_img));
}

TEST(UtilsTest, cuda_sobel_kernel)
{
    Eigen::MatrixXd cpu_sobel_kernel_x = sobel_x_kernel();
    cuda::CuMatrixD cuda_sobel_kernel_x = cuda::sobel_x_kernel();
    auto gpu_copied_kernel = cuda::cuda_to_eigen(cuda_sobel_kernel_x);
    EXPECT_TRUE(are_matrices_close(gpu_copied_kernel, cpu_sobel_kernel_x));
}

void cuda_correlation_test(size_t kernel_size, size_t n_rows, size_t n_cols, bool debug)
{
    size_t patch_size = kernel_size;
    Eigen::MatrixXd kernel = Eigen::MatrixXd::Random(patch_size, patch_size);
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(n_rows, n_cols);
    auto correlated = correlation(matrix, kernel);
    cuda::CuMatrixD cuda_kernel = cuda::eigen_to_cuda(kernel);
    cuda::CuMatrixD cuda_matrix = cuda::eigen_to_cuda(matrix);
    cuda::CuMatrixD cuda_correlated = cuda::correlation(cuda_matrix, cuda_kernel);
    auto cuda_correlated_cpu = cuda::cuda_to_eigen(cuda_correlated);

    size_t s = patch_size/2;
    size_t l = std::min(matrix.cols(), matrix.rows()) - (2 * s);
    if (debug)
    {
        std::cout << s << " " << l << std::endl;
        std::cout << "cpu correlated:\n"
                    << correlated << std::endl;
        std::cout << "gpu correlated:\n"
                    << cuda_correlated_cpu << std::endl;
    }
    EXPECT_TRUE(are_matrices_close(
        cuda_correlated_cpu.block(s,s,l,l),
        correlated.block(s,s,l,l)));
    
}

TEST(UtilsTest, cuda_correlation)
{
    for (int i = 0; i < 10; ++i)
    {
        cuda_correlation_test(3, 5, 6, false);
        cuda_correlation_test(3, 5, 5, false);
    }
}

TEST(UtilsTest, cuda_ew_square)
{
    for (int i = 0; i< 100; ++i)
    {
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(500, 500);
        auto cpu_squared = matrix.array().square().matrix();
        cuda::CuMatrixD cuda_matrix = cuda::eigen_to_cuda(matrix);
        auto squared = cuda::pow(cuda_matrix,2.0);                              // out of place squared
        auto in_place_cuda_squared = cuda::pow(std::move(cuda_matrix), 2.0);    // in place squared
        EXPECT_EQ(in_place_cuda_squared.d_data.get(), cuda_matrix.d_data.get());
        auto gpu_squared = cuda::cuda_to_eigen(squared);
        auto gpu_inplace_squared = cuda::cuda_to_eigen(in_place_cuda_squared);
        EXPECT_TRUE(are_matrices_close(gpu_squared, cpu_squared));
        EXPECT_TRUE(are_matrices_close(gpu_inplace_squared, cpu_squared));
    }
}

TEST(UtilsTest, ew_multiplication)
{
    for (int i = 0; i< 10; ++i)
    {
        Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(500, 500);
        Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(500, 500);
        auto cpu_ew_product = (m1.array() * m2.array()).matrix();
        cuda::CuMatrixD cuda_m1 = cuda::eigen_to_cuda(m1);
        cuda::CuMatrixD cuda_m2 = cuda::eigen_to_cuda(m2);
        auto cuda_product = cuda_m1 * cuda_m2;
        auto cuda_product_on_cpu = cuda::cuda_to_eigen(cuda_product);
        EXPECT_TRUE(are_matrices_close(cuda_product_on_cpu, cpu_ew_product));
        auto res_1 = (cuda_m1 * cuda_m2) * cuda_m1; 
        auto res_2 = (cuda_m1 * cuda_m2) * (cuda_m1 * cuda_m2);
        auto res_3 = cuda_m2 * (cuda_m1 * cuda_m2);
        // std::cout << "cpu squared\n" << cpu_squared << std::endl;
        // std::cout << "gpu squared\n" << gpu_squared << std::endl;
    }
}


#endif