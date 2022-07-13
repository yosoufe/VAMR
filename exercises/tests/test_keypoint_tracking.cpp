#include "keypoint_tracking.hpp"
#include <gtest/gtest.h>

TEST(keypoint_tracking, calculate_Is)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Ones(3, 3);
    Eigen::MatrixXd sI_xx, sI_yy, sI_xy;

    input << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;
    calculate_Is(input, 3, sI_xx, sI_yy, sI_xy);

    Eigen::MatrixXd expected_I = Eigen::MatrixXd::Zero(3, 3);
    expected_I(1, 1) = 8 * 8;
    EXPECT_TRUE(are_matrices_close(expected_I, sI_xx));

    expected_I(1, 1) = 24 * 24;
    EXPECT_TRUE(are_matrices_close(expected_I, sI_yy));

    expected_I(1, 1) = 24 * 8;
    EXPECT_TRUE(are_matrices_close(expected_I, sI_xy));
}

#if WITH_CUDA

TEST(keypoint_tracking, calculate_Is_gpu)
{
    for (int i = 0; i < 10; ++i)
    {
        Eigen::MatrixXd h_img = Eigen::MatrixXd::Random(15, 16);
        cuda::CuMatrixD d_img = cuda::eigen_to_cuda(h_img);
        Eigen::MatrixXd h_sI_xx, h_sI_yy, h_sI_xy;
        cuda::CuMatrixD d_sI_xx, d_sI_yy, d_sI_xy;
        size_t patch_size = 9;
        calculate_Is(h_img, patch_size, h_sI_xx, h_sI_yy, h_sI_xy);
        cuda::calculate_Is(d_img, patch_size, d_sI_xx, d_sI_yy, d_sI_xy);
        auto hd_sI_xx = cuda::cuda_to_eigen(d_sI_xx);
        auto hd_sI_yy = cuda::cuda_to_eigen(d_sI_yy);
        auto hd_sI_xy = cuda::cuda_to_eigen(d_sI_xy);

        EXPECT_TRUE(are_matrices_close(hd_sI_xx,
                                       h_sI_xx));
        EXPECT_TRUE(are_matrices_close(hd_sI_yy,
                                       h_sI_yy));
        EXPECT_TRUE(are_matrices_close(hd_sI_xy,
                                       h_sI_xy));
        // std::cout << "sI_xx CPU\n"
        //           << h_sI_xx << std::endl<< std::endl;
        // std::cout << "sI_xx GPU\n"
        //           << hd_sI_xx << std::endl;
    }
}

TEST(keypoint_tracking, harris_score)
{
    for (int i = 0; i < 10; ++i)
    {
        Eigen::MatrixXd h_img = Eigen::MatrixXd::Random(10, 10);
        cuda::CuMatrixD d_img = cuda::eigen_to_cuda(h_img);
        size_t patch_size = 5;
        double kappa = 0.08;
        auto h_score = harris(h_img, patch_size, kappa);
        auto d_score = harris(d_img, patch_size, kappa);
        auto hd_score = cuda::cuda_to_eigen(d_score);
        EXPECT_TRUE(are_matrices_close(hd_score,
                                       h_score));

        // std::cout << "h_score CPU\n"
        //           << h_score << std::endl;
        // std::cout << "d_score GPU\n"
        //           << hd_score << std::endl;
    }
}

TEST(keypoint_tracking, shi_tomasi_score)
{
    for (int i = 0; i < 100; ++i)
    {
        Eigen::MatrixXd h_img = Eigen::MatrixXd::Random(10, 10);
        cuda::CuMatrixD d_img = cuda::eigen_to_cuda(h_img);
        size_t patch_size = 5;
        auto h_score = shi_tomasi(h_img, patch_size);
        auto d_score = shi_tomasi(d_img, patch_size);
        auto hd_score = cuda::cuda_to_eigen(d_score);
        EXPECT_TRUE(are_matrices_close(hd_score,
                                       h_score));

        // std::cout << "h_score CPU\n"
        //           << h_score << std::endl;
        // std::cout << "d_score GPU\n"
        //           << hd_score << std::endl;
    }
}

TEST(keypoint_tracking, non_maximum_suppression)
{
    Eigen::MatrixXd input(4, 5);
    input << 10, 1, 3, 6, 10,
        2, 3, 7, 3, 1,
        5, 4, 4, 1, 2,
        4, 1, 9, 0, 3;
    int patch_size = 3;

    Eigen::MatrixXd expected(4, 5);
    expected << 10, 0, 0, 0, 10,
        0, 0, 7, 0, 0,
        5, 0, 0, 0, 0,
        0, 0, 9, 0, 3;

    auto d_input = cuda::eigen_to_cuda(input);
    auto d_output = cuda::non_maximum_suppression(d_input, patch_size);
    auto hd_output = cuda::cuda_to_eigen(d_output);
    EXPECT_TRUE(are_matrices_close(hd_output, expected));
    // std::cout << hd_output << std::endl;
}

#endif