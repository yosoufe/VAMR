#include "keypoint_tracking.hpp"
#include <cassert>
#include "utils.cuh"
#include "operations.hpp"
#include "utils.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

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

__global__ void non_maximum_suppression_kernel_1(double *input, int n_rows, int n_cols, int patch_size, double *output)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row >= n_rows || col >= n_cols)
        return;

    int radius = patch_size / 2;

    auto &center = input[get_index_colwise(row, col, n_rows)];
    auto &output_el = output[get_index_colwise(row, col, n_rows)];

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

            auto &other = input[get_index_colwise(other_row, other_col, n_rows)];
            if (center < other)
            {
                output_el = 0;
                return;
            }
        }
    }
    output[get_index_colwise(row, col, n_rows)] = input[get_index_colwise(row, col, n_rows)];
}

cuda::CuMatrixD cuda::non_maximum_suppression_1(const cuda::CuMatrixD &input, size_t patch_size)
{
    assert(patch_size % 2 == 1);
    int num_thread_1d = 32;
    dim3 grid_dim(input.n_rows / num_thread_1d + 1, input.n_cols / num_thread_1d + 1);
    dim3 block_dim(num_thread_1d, num_thread_1d);
    cuda::CuMatrixD output(input.n_cols, input.n_rows);
    non_maximum_suppression_kernel_1<<<grid_dim, block_dim>>>(input.d_data.get(),
                                                              input.n_rows,
                                                              input.n_cols,
                                                              patch_size,
                                                              output.d_data.get());
    CLE();
    CSC(cudaDeviceSynchronize());
    return output;
}

__global__ void non_maximum_suppression_kernel_2(double *input, int n_rows, int n_cols, int patch_size, double *output)
{
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ double shared[];

    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row >= n_rows || col >= n_cols)
        return;

    // indicies in shared memory
    int radius = patch_size / 2;
    int sh_row_zero = blockDim.x * blockIdx.x - radius;
    int sh_col_zero = blockDim.y * blockIdx.y - radius;

    int n_rows_shared = blockDim.x + patch_size - 1;

    // copy to shared memory
    for (int g_row = row - radius; g_row <= row + radius; ++g_row)
    {
        int sh_row = g_row - sh_row_zero;
        for (int g_col = col - radius; g_col <= col + radius; ++g_col)
        {
            int sh_col = g_col - sh_col_zero;
            int shared_idx = get_index_colwise(sh_row, sh_col, n_rows_shared);
            if (g_col < 0 || g_col >= n_cols || g_row < 0 || g_row >= n_rows)
                shared[shared_idx] = 0; // set to zero if corresponding global is out of range
            else
                shared[shared_idx] = input[get_index_colwise(g_row, g_col, n_rows)];
        }
    }

    cg::sync(cta);

    int sh_row = row - sh_row_zero;
    int sh_col = col - sh_col_zero;

    auto &center = shared[get_index_colwise(sh_row, sh_col, n_rows_shared)];
    auto &output_el = output[get_index_colwise(row, col, n_rows)];

    for (int other_row = sh_row - radius; other_row <= sh_row + radius; ++other_row)
    {
        // if (other_row < 0 || other_row >= n_rows)
        //     continue;

        for (int other_col = sh_col - radius; other_col <= sh_col + radius; ++other_col)
        {
            // This is not required since shared memory is always initiailzied
            // if (other_col < 0 || other_col > n_cols)
            //     continue;

            if (other_col == sh_col && other_row == sh_row)
                continue;

            auto &other = shared[get_index_colwise(other_row, other_col, n_rows_shared)];
            if (center < other)
            {
                output_el = 0;
                return;
            }
        }
    }
    output_el = center;
}

cuda::CuMatrixD cuda::non_maximum_suppression_2(const cuda::CuMatrixD &input, size_t patch_size)
{
    assert(patch_size % 2 == 1);
    int num_thread_1d = 32;
    dim3 grid_dim(input.n_rows / num_thread_1d + 1, input.n_cols / num_thread_1d + 1);
    dim3 block_dim(num_thread_1d, num_thread_1d);

    cuda::CuMatrixD output(input.n_cols, input.n_rows);
    int shared_mem_size = ::pow(num_thread_1d + patch_size - 1, 2) * sizeof(double);
    non_maximum_suppression_kernel_2<<<grid_dim, block_dim, shared_mem_size>>>(input.d_data.get(),
                                                                               input.n_rows,
                                                                               input.n_cols,
                                                                               patch_size,
                                                                               output.d_data.get());
    CLE();
    CSC(cudaDeviceSynchronize());
    return output;
}

__global__ void non_maximum_suppression_kernel_3(double *input, int n_rows, int n_cols, int patch_size, double *output)
{
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ double shared[];

    int halo_size = patch_size - 1;
    int halo_half = patch_size / 2;
    int row = (threadIdx.x - halo_half) + (blockDim.x - halo_size) * blockIdx.x;
    int col = (threadIdx.y - halo_half) + (blockDim.y - halo_size) * blockIdx.y;

    if (row >= n_rows || col >= n_cols)
        return;

    // indicies in shared memory
    int radius = patch_size / 2;
    int sh_row = threadIdx.x;
    int sh_col = threadIdx.y;

    int n_rows_shared = blockDim.x;

    int shared_idx = get_index_colwise(sh_row, sh_col, n_rows_shared);

    int g_row = blockIdx.x * (blockDim.x - halo_size) + threadIdx.x;
    int g_col = blockIdx.y * (blockDim.y - halo_size) + threadIdx.y;

    if (g_col < 0 || g_col >= n_cols || g_row < 0 || g_row >= n_rows)
        shared[shared_idx] = 0; // set to zero if corresponding global is out of range
    else
        shared[shared_idx] = input[get_index_colwise(g_row, g_col, n_rows)];

    cg::sync(cta);

    if (sh_row < halo_half || sh_col < halo_half || sh_row >= blockDim.x - halo_half || sh_col >= blockDim.y - halo_half)
        return;


    auto &center = shared[get_index_colwise(sh_row, sh_col, n_rows_shared)];
    int output_idx = get_index_colwise(row, col, n_rows);

    for (int other_row = sh_row - radius; other_row <= sh_row + radius; ++other_row)
    {
        // if (other_row < 0 || other_row >= n_rows)
        //     continue;

        for (int other_col = sh_col - radius; other_col <= sh_col + radius; ++other_col)
        {
            // This is not required since shared memory is always initiailzied
            // if (other_col < 0 || other_col > n_cols)
            //     continue;

            if (other_col == sh_col && other_row == sh_row)
                continue;

            auto &other = shared[get_index_colwise(other_row, other_col, n_rows_shared)];
            if (center < other)
            {
                output[output_idx] = 0;
                return;
            }
        }
    }
    output[output_idx] = center;
}

cuda::CuMatrixD cuda::non_maximum_suppression_3(const cuda::CuMatrixD &input, size_t patch_size)
{
    assert(patch_size % 2 == 1);
    int halo_size = patch_size - 1;

    int num_thread_1d = 32;
    dim3 grid_dim((input.n_rows + halo_size) / num_thread_1d + 1,
                  (input.n_cols + halo_size) / num_thread_1d + 1);
    dim3 block_dim(num_thread_1d,
                   num_thread_1d);

    cuda::CuMatrixD output(input.n_cols, input.n_rows);
    int shared_mem_size = ::pow(num_thread_1d + patch_size - 1, 2) * sizeof(double);
    non_maximum_suppression_kernel_2<<<grid_dim, block_dim, shared_mem_size>>>(input.d_data.get(),
                                                                               input.n_rows,
                                                                               input.n_cols,
                                                                               patch_size,
                                                                               output.d_data.get());
    CLE();
    CSC(cudaDeviceSynchronize());
    return output;
}