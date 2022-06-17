#include "keypoint_tracking.hpp"
#include "utils.cuh"
#include "operations.hpp"

void cuda::calculate_Is(
    const cuda::CuMatrixD &img,
    size_t patch_size,
    cuda::CuMatrixD &sI_xx,
    cuda::CuMatrixD &sI_yy,
    cuda::CuMatrixD &sI_xy)
{
    // TODO: test it.
    // TODO: generalize the correlation to different kernel sizes.
    auto I_xx = cuda::correlation(img, cuda::sobel_x_kernel());
    auto I_yy = cuda::correlation(img, cuda::sobel_y_kernel());
    auto I_xy = cuda::ew_multiplication(I_xx, I_yy);
    ew_square(I_xx);
    ew_square(I_yy);
    auto ones = cuda::ones(patch_size, patch_size);
    sI_xx = cuda::correlation(I_xx, ones);
    sI_yy = cuda::correlation(I_yy, ones);
    sI_xy = cuda::correlation(I_xy, ones);
    I_xx.free(); I_xy.free(); I_yy.free(); ones.free();
}

cuda::CuMatrixD cuda::harris(const cuda::CuMatrixD &img, size_t patch_size, double kappa)
{
    cuda::CuMatrixD sI_xx, sI_yy, sI_xy;
    calculate_Is(img, patch_size, sI_xx, sI_yy, sI_xy);
    // calculate score;
    // threshold the score;
    // return the score;
    return cuda::CuMatrixD();
}

cuda::CuMatrixD cuda::shi_tomasi(const cuda::CuMatrixD &img, size_t patch_size)
{
    return cuda::CuMatrixD();
}