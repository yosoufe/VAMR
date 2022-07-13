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
    cuda::CuMatrixD score = (sI_xx * sI_yy - 2 * std::move(sI_xy)) - kappa * cuda::pow((sI_xx + sI_yy), 2);
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

__global__ void non_maximum_suppression_kernel(double *input, int n_rows, int n_cols, int patch_size, double *output)
{
    int row = threadIdx.x;
    int col = threadIdx.y;

    if (row >= n_rows || col >= n_cols)
        return;

    int radius = patch_size / 2;
    
    auto & center = input[get_index_colwise(row, col, n_rows)];
    auto & output_el = output[get_index_colwise(row, col, n_rows)];

    for (int other_row = row - radius; other_row <= row + radius; ++other_row)
    {
        if (other_row < 0 || other_row >= n_rows)
            continue;

        for (int other_col = col - radius; other_col <= col + radius; ++other_col)
        {
            
            if (other_col < 0 || other_col > n_cols)
                continue;

            if (other_col == col && other_row == row)
                continue;

            auto & other = input[get_index_colwise(other_row, other_col, n_rows)];
            if (center < other)
            {
                output_el = 0;
                return;
            }
        }
    }
    output[get_index_colwise(row, col, n_rows)] = input[get_index_colwise(row, col, n_rows)];
}

cuda::CuMatrixD cuda::non_maximum_suppression(const cuda::CuMatrixD &input, size_t patch_size)
{
    dim3 blocks(1, 1);
    dim3 threads(input.n_rows, input.n_cols);
    cuda::CuMatrixD output(input.n_cols, input.n_rows);
    non_maximum_suppression_kernel<<<blocks, threads>>>(input.d_data.get(), input.n_rows, input.n_cols, patch_size, output.d_data.get());
    CLE();
    CSC(cudaDeviceSynchronize());
    return output;
}