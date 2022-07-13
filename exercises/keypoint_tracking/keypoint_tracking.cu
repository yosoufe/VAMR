#include "keypoint_tracking.hpp"
#include "utils.cuh"
#include "operations.hpp"
#include "utils.cuh"

void cuda::calculate_Is(
    const cuda::CuMatrixD &img,
    size_t patch_size,
    cuda::CuMatrixD &sI_xx,
    cuda::CuMatrixD &sI_yy,
    cuda::CuMatrixD &sI_xy)
{
    auto I_xx = cuda::correlation(img, cuda::sobel_x_kernel());
    auto I_yy = cuda::correlation(img, cuda::sobel_y_kernel());
    auto I_xy = I_xx * I_yy;
    I_xx = cuda::pow(std::move(I_xx), 2.0);
    I_yy = cuda::pow(std::move(I_yy), 2.0);
    auto ones = cuda::ones(patch_size, patch_size);
    sI_xx = cuda::correlation(I_xx, ones);
    sI_yy = cuda::correlation(I_yy, ones);
    sI_xy = cuda::correlation(I_xy, ones);
}

cuda::CuMatrixD cuda::harris(const cuda::CuMatrixD &img, size_t patch_size, double kappa)
{
    cuda::CuMatrixD sI_xx, sI_yy, sI_xy;
    cuda::calculate_Is(img, patch_size, sI_xx, sI_yy, sI_xy);
    
    // calculate score;
    cuda::CuMatrixD score = (sI_xx * sI_yy - 2 * std::move(sI_xy)) - kappa * cuda::pow((sI_xx + sI_yy),2);
    score = threshold_lower(std::move(score), 0, 0);
    return score;
}

cuda::CuMatrixD cuda::shi_tomasi(const cuda::CuMatrixD &img, size_t patch_size)
{
    cuda::CuMatrixD sI_xx, sI_yy, sI_xy;
    cuda::calculate_Is(img, patch_size, sI_xx, sI_yy, sI_xy);
    auto trace = sI_xx + sI_yy;
    auto determinant = sI_xx * sI_yy - cuda::pow(sI_xy, 2);
    auto score = (trace * 0.5 - cuda::pow(cuda::pow(trace * 0.5, 2) - determinant, 0.5));
    score = threshold_lower(std::move(score), 0, 0);
    return score;
}

cuda::CuMatrixD cuda::non_maximum_suppression(const cuda::CuMatrixD &input, size_t patch_size)
{
    
}