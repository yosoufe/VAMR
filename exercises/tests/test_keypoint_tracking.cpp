#include "keypoint_tracking.hpp"
#include <gtest/gtest.h>

#if WITH_CUDA

TEST(keypoint_tracking, calculate_Is)
{
    for (int i = 0; i < 100 ; ++i)
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
        size_t s = 1+patch_size/2;
        size_t l = h_img.cols()-(2*s);
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
        d_img.free();
        d_sI_xx.free(); d_sI_yy.free(); d_sI_xy.free();
    }
}

#endif