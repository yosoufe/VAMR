#include "keypoint_tracking.hpp"
#include "utils.cuh"

using namespace cuda;

CuMatrixD cuda::harris(const CuMatrixD &img, size_t patch_size, double kappa)
{
    // calculate_Is();
    // calculate score;
    // threshold the score;
    // return the score;
    return CuMatrixD();
}

CuMatrixD cuda::shi_tomasi(const CuMatrixD &img, size_t patch_size)
{
    return CuMatrixD();
}