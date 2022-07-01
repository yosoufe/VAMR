#include "keypoint_tracking.hpp"
#include <gtest/gtest.h>

#if WITH_CUDA

TEST(keypoint_tracking, calculate_Is_gpu)
{
    for (int i = 0; i < 100; ++i)
    {
        Eigen::MatrixXd h_img = Eigen::MatrixXd::Random(10, 10);
        cuda::CuMatrixD d_img = cuda::eigen_to_cuda(h_img);
        Eigen::MatrixXd h_sI_xx, h_sI_yy, h_sI_xy;
        cuda::CuMatrixD d_sI_xx, d_sI_yy, d_sI_xy;
        size_t patch_size = 5;
        calculate_Is(h_img, patch_size, h_sI_xx, h_sI_yy, h_sI_xy);
        calculate_Is(d_img, patch_size, d_sI_xx, d_sI_yy, d_sI_xy);
        auto hd_sI_xx = cuda::cuda_to_eigen(d_sI_xx);
        auto hd_sI_yy = cuda::cuda_to_eigen(d_sI_yy);
        auto hd_sI_xy = cuda::cuda_to_eigen(d_sI_xy);
        size_t s = 1 + patch_size / 2;
        size_t l = h_img.cols() - (2 * s);
        EXPECT_TRUE(are_matrices_close(hd_sI_xx.block(s, s, l, l),
                                       h_sI_xx.block(s, s, l, l)));
        EXPECT_TRUE(are_matrices_close(hd_sI_yy.block(s, s, l, l),
                                       h_sI_yy.block(s, s, l, l)));
        EXPECT_TRUE(are_matrices_close(hd_sI_xy.block(s, s, l, l),
                                       h_sI_xy.block(s, s, l, l)));
        // std::cout << "sI_xx CPU\n"
        //           << h_sI_xx << std::endl;
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