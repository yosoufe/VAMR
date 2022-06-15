#include "keypoint_tracking.hpp"
#include "utils.cuh"
#include "operations.hpp"

void calculate_Is(
    const cuda::CuMatrixD &img,
    size_t patch_size,
    cuda::CuMatrixD &sI_xx,
    cuda::CuMatrixD &sI_yy,
    cuda::CuMatrixD &sI_xy)
{
    // CUDA TODO
    auto I_xx = cuda::correlation(img, cuda::sobel_x_kernel());
    auto I_yy = cuda::correlation(img, cuda::sobel_y_kernel());
    ew_square(I_xx);
    ew_square(I_yy);
    auto I_xy = (I_x.array() * I_y.array()).matrix();
    sI_xx = cuda::correlation(I_xx, Eigen::MatrixXd::Ones(patch_size, patch_size));
    sI_yy = cuda::correlation(I_yy, Eigen::MatrixXd::Ones(patch_size, patch_size));
    sI_xy = cuda::correlation(I_xy, Eigen::MatrixXd::Ones(patch_size, patch_size));
}

cuda::CuMatrixD cuda::harris(const cuda::CuMatrixD &img, size_t patch_size, double kappa)
{
    // calculate_Is();
    // calculate score;
    // threshold the score;
    // return the score;
    return cuda::CuMatrixD();
}

cuda::CuMatrixD cuda::shi_tomasi(const cuda::CuMatrixD &img, size_t patch_size)
{
    return cuda::CuMatrixD();
}