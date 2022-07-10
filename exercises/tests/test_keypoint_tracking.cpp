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

    Eigen::MatrixXd expected_I = Eigen::MatrixXd::Zero(3,3);
    expected_I(1,1) = 8*8;
    EXPECT_TRUE(are_matrices_close(expected_I, sI_xx));

    expected_I(1,1) = 24*24;
    EXPECT_TRUE(are_matrices_close(expected_I, sI_yy));

    expected_I(1,1) = 24*8;
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
        size_t s = 1 + patch_size / 2;
        size_t l = std::min(h_img.cols(), h_img.rows()) - (2 * s);
        EXPECT_TRUE(are_matrices_close(hd_sI_xx.block(s, s, l, l),
                                       h_sI_xx.block(s, s, l, l)));
        EXPECT_TRUE(are_matrices_close(hd_sI_yy.block(s, s, l, l),
                                       h_sI_yy.block(s, s, l, l)));
        EXPECT_TRUE(are_matrices_close(hd_sI_xy.block(s, s, l, l),
                                       h_sI_xy.block(s, s, l, l)));
        // std::cout << s << " " << l << std::endl;
        // std::cout << "sI_xx CPU\n"
        //           << h_sI_xx << std::endl<< std::endl;
        // std::cout << "sI_xx GPU\n"
        //           << hd_sI_xx << std::endl;
    }
}

TEST(keypoint_tracking, harris_score)
{
    for (int i = 0; i < 100; ++i)
    {
        Eigen::MatrixXd h_img = Eigen::MatrixXd::Random(10, 10);
        cuda::CuMatrixD d_img = cuda::eigen_to_cuda(h_img);
        size_t patch_size = 5;
        double kappa = 0.08;
        auto h_score = harris(h_img, patch_size, kappa);
        auto d_score = harris(d_img, patch_size, kappa);
        auto hd_score = cuda::cuda_to_eigen(d_score);
        size_t s = 1 + patch_size / 2;
        size_t l = h_img.cols() - (2 * s);
        EXPECT_TRUE(are_matrices_close(hd_score.block(s, s, l, l),
                                       h_score.block(s, s, l, l)));

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
        size_t s = 1 + patch_size / 2;
        size_t l = h_img.cols() - (2 * s);
        EXPECT_TRUE(are_matrices_close(hd_score.block(s, s, l, l),
                                       h_score.block(s, s, l, l)));

        // std::cout << "h_score CPU\n"
        //           << h_score << std::endl;
        // std::cout << "d_score GPU\n"
        //           << hd_score << std::endl;
    }
}

#endif