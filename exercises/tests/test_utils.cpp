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
        EXPECT_EQ(cuda_eigen_double.cols(), double_matrix.cols());
        EXPECT_EQ(cuda_eigen_double.rows(), double_matrix.rows());

        auto new_double_matrix = cuda::cuda_to_eigen(cuda_eigen_double);
        EXPECT_EQ(new_double_matrix.cols(), double_matrix.cols());
        EXPECT_EQ(new_double_matrix.rows(), double_matrix.rows());
        EXPECT_NEAR((new_double_matrix - double_matrix).norm(), 0.0, 1e-5);

        Eigen::MatrixXf float_matrix = Eigen::MatrixXf::Random(5, 5);
        auto cuda_eigen_float = cuda::eigen_to_cuda(float_matrix);
        EXPECT_EQ(cuda_eigen_float.cols(), float_matrix.cols());
        EXPECT_EQ(cuda_eigen_float.rows(), float_matrix.rows());

        auto new_float_matrix = cuda::cuda_to_eigen(cuda_eigen_float);
        EXPECT_TRUE(are_matrices_close(new_float_matrix, float_matrix));
    }
}

void test_cuda_block(int n_rows_original,
                     int n_cols_original,
                     int start_row,
                     int start_col,
                     int block_height,
                     int block_width)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(n_rows_original, n_cols_original);
    Eigen::MatrixXd expected = input.block(start_row, start_col, block_height, block_width);

    auto d_input = cuda::eigen_to_cuda(input);
    auto d_output = d_input.block(start_row, start_col, block_height, block_width);
    EXPECT_TRUE(are_matrices_close(d_output, expected));

    // std::cout << "input\n"
    //           << input << std::endl;
    // std::cout << "expected\n"
    //           << expected << std::endl;
    // std::cout << "output\n"
    //           << cuda::cuda_to_eigen(d_output) << std::endl;
}

TEST(UtilsTest, cuda_block)
{
    for (int i = 0; i < 3; ++i)
    {
        auto start = second();
        test_cuda_block(10, 9, 1, 1, 8, 8);
        std::cout << "time (8,8): " << second() - start << std::endl;
    }

    for (int i = 0; i < 3; ++i)
    {
        auto start = second();
        test_cuda_block(1000, 1000, 1, 1, 1000-1, 1000-1);
        std::cout << "time (1000,1000): " << second() - start << std::endl;
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

    if (debug)
    {
        std::cout << "cpu correlated:\n"
                  << correlated << std::endl;
        std::cout << "gpu correlated:\n"
                  << cuda_correlated_cpu << std::endl;
    }
    EXPECT_TRUE(are_matrices_close(
        cuda_correlated_cpu,
        correlated));
}

TEST(UtilsTest, cuda_correlation)
{
    for (int i = 0; i < 1; ++i)
    {
        cuda_correlation_test(3, 5, 6, false);
        cuda_correlation_test(3, 5, 5, false);
    }
}

TEST(UtilsTest, cuda_ew_square)
{
    setup_back_track();
    for (int i = 0; i < 100; ++i)
    {
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(500, 500);
        auto cpu_squared = matrix.array().square().matrix();
        cuda::CuMatrixD cuda_matrix = cuda::eigen_to_cuda(matrix);
        auto squared = cuda::pow(cuda_matrix, 2.0);                          // out of place squared
        auto in_place_cuda_squared = cuda::pow(std::move(cuda_matrix), 2.0); // in place squared
        EXPECT_EQ(in_place_cuda_squared.data(), cuda_matrix.data());
        auto gpu_squared = cuda::cuda_to_eigen(squared);
        auto gpu_inplace_squared = cuda::cuda_to_eigen(in_place_cuda_squared);
        EXPECT_TRUE(are_matrices_close(gpu_squared, cpu_squared));
        EXPECT_TRUE(are_matrices_close(gpu_inplace_squared, cpu_squared));
    }
}

TEST(UtilsTest, cuda_norm)
{
    for (int i = 0; i < 100; ++i)
    {
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(500, 1);
        double h_norm = matrix.norm();
        cuda::CuMatrixD d_matrix = cuda::eigen_to_cuda(matrix);
        auto d_norm = cuda::norm(d_matrix);
        EXPECT_NEAR(h_norm, d_norm, 1e-10);
    }
}

TEST(UtilsTest, ew_multiplication)
{
    for (int i = 0; i < 10; ++i)
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

TEST(UtilsTest, cuda_zero_borders)
{
    size_t rows = 10, cols = 8;
    size_t s_row = 2, s_col = 1, l_row = 5, l_col = 7;
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(rows, cols);
    Eigen::MatrixXd expected_output = Eigen::MatrixXd::Zero(rows, cols);
    expected_output.block(s_row, s_col, l_row, l_col) = input.block(s_row, s_col, l_row, l_col);

    auto d_input = cuda::eigen_to_cuda(input);
    cuda::zero_borders(d_input, s_row, s_col, l_row, l_col);
    auto hd_output = cuda::cuda_to_eigen(d_input);
    EXPECT_TRUE(are_matrices_close(hd_output, expected_output));

    // printf("s_row %ld, s_col %ld, l_row %ld, l_col %ld.\n", s_row, s_col, l_row, l_col);
    // std::cout << "input\n" << input << std::endl;
    // std::cout << "output\n" << hd_output << std::endl;
}

#endif