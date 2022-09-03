#include "keypoint_tracking.hpp"
#include <cassert>
#include "utils.cuh"
#include "operations.hpp"
#include "utils.cuh"
#include "operations.cuh"
#include <cooperative_groups.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_malloc.h>

#include <cub/cub.cuh>

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
    dim3 grid_dim(input.rows() / num_thread_1d + 1, input.cols() / num_thread_1d + 1);
    dim3 block_dim(num_thread_1d, num_thread_1d);
    cuda::CuMatrixD output(input.rows(), input.cols());
    non_maximum_suppression_kernel_1<<<grid_dim, block_dim>>>(input.data(),
                                                              input.rows(),
                                                              input.cols(),
                                                              patch_size,
                                                              output.data());
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
    dim3 grid_dim(input.rows() / num_thread_1d + 1, input.cols() / num_thread_1d + 1);
    dim3 block_dim(num_thread_1d, num_thread_1d);

    cuda::CuMatrixD output(input.rows(), input.cols());
    int shared_mem_size = ::pow(num_thread_1d + patch_size - 1, 2) * sizeof(double);
    non_maximum_suppression_kernel_2<<<grid_dim, block_dim, shared_mem_size>>>(input.data(),
                                                                               input.rows(),
                                                                               input.cols(),
                                                                               patch_size,
                                                                               output.data());
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
    dim3 grid_dim((input.rows() + halo_size) / num_thread_1d + 1,
                  (input.cols() + halo_size) / num_thread_1d + 1);
    dim3 block_dim(num_thread_1d,
                   num_thread_1d);

    cuda::CuMatrixD output(input.rows(), input.cols());
    int shared_mem_size = ::pow(num_thread_1d + patch_size - 1, 2) * sizeof(double);
    non_maximum_suppression_kernel_2<<<grid_dim, block_dim, shared_mem_size>>>(input.data(),
                                                                               input.rows(),
                                                                               input.cols(),
                                                                               patch_size,
                                                                               output.data());
    CLE();
    CSC(cudaDeviceSynchronize());
    return output;
}

__global__ void index_conversion(int *idx, double *output, int n_rows, int n_cols)
{
    int index_1d = threadIdx.x + blockDim.x * blockIdx.x;
    if (index_1d >= n_rows * n_cols)
        return;

    auto res = get_2d_index_colwise(idx[index_1d], n_rows);
    auto row = thrust::get<0>(res);
    auto col = thrust::get<1>(res);

    auto res_row_index = get_index_colwise(0, index_1d, 2);
    auto res_col_index = get_index_colwise(1, index_1d, 2);

    output[res_row_index] = col;
    output[res_col_index] = row;
}

void _sort_matrix(cuda::CuMatrixD &input,
                  cuda::CuMatrixD &indicies_output)
{
    auto indices = cuda::create_indices(input);
    auto key_start = thrust::device_pointer_cast(input.data());
    auto key_end = key_start + input.n_elements();
    thrust::sort_by_key(thrust::cuda::par, key_start, key_end, indices, thrust::greater<double>());

    indicies_output = cuda::CuMatrixD(2, input.n_elements());
    index_conversion<<<input.n_elements() / 1024 + 1, 1024>>>(indices.get(),
                                                              indicies_output.data(),
                                                              input.rows(),
                                                              input.cols());
}

cuda::CuMatrixD cuda::sort_matrix(
    cuda::CuMatrixD &&input,
    cuda::CuMatrixD &indicies_output)
{
    _sort_matrix(input, indicies_output);
    return input;
}

cuda::CuMatrixD cuda::sort_matrix(
    const cuda::CuMatrixD &input,
    cuda::CuMatrixD &indicies_output)
{
    auto output_values = input.clone();
    _sort_matrix(output_values, indicies_output);
    return output_values;
}

cuda::CuMatrixD cuda::select_keypoints(
    const cuda::CuMatrixD &score,
    size_t radius)
{
    int patch_size = radius * 2 + 1;
    cuda::CuMatrixD indicies;

    cuda::sort_matrix(cuda::non_maximum_suppression_1(score, patch_size), indicies);
    // keys are scores and values are indicies.
    return indicies;
}

__global__ void
describe_keypoints_kernel(double *output,
                          double *img,
                          double *kps, int num_kp,
                          int radius,
                          int img_rows, int img_cols)
{
    // read coordinates from kps
    int des_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (des_idx >= num_kp)
        return;

    // read coordiantes
    int col = kps[get_index_colwise(0, des_idx, 2)];
    int row = kps[get_index_colwise(1, des_idx, 2)];
    int desc_length = (2 * radius + 1) * (2 * radius + 1);

    // loop over image around the coordinate within radius
    int counter = 0;
    for (int col_to_copy = col - radius; col_to_copy <= col + radius; ++col_to_copy)
    {
        for (int row_to_copy = row - radius; row_to_copy <= row + radius; ++row_to_copy)
        {
            // copy values to output descriptors
            auto output_idx = get_index_colwise(counter++, des_idx, desc_length);
            if (row_to_copy < 0 || row_to_copy >= img_rows ||
                col_to_copy < 0 || col_to_copy >= img_cols)
            {
                output[output_idx] = 0;
            }
            else
            {
                output[output_idx] = img[get_index_colwise(row_to_copy, col_to_copy, img_rows)];
            }
        }
    }
}

cuda::CuMatrixD cuda::describe_keypoints(
    const cuda::CuMatrixD &img,
    const cuda::CuMatrixD &sorted_pixels_based_on_scores,
    int num_keypoints_to_consider,
    int descriptor_radius)
{
    int descriptor_length = ::pow(2 * descriptor_radius + 1, 2);
    cuda::CuMatrixD output(descriptor_length, num_keypoints_to_consider);

    dim3 block_dim(min(1024, num_keypoints_to_consider));
    dim3 grid_dim(num_keypoints_to_consider / block_dim.x + 1);

    describe_keypoints_kernel<<<grid_dim, block_dim>>>(
        output.data(),
        img.data(),
        sorted_pixels_based_on_scores.data(),
        num_keypoints_to_consider,
        descriptor_radius,
        img.rows(),
        img.cols());

    return output;
}

__global__ void
calculate_columnwise_sum_kernel(
    double *output,
    double *input,
    int num_rows,
    int num_cols)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (col >= num_cols)
        return;
    output[col] = 0;
    double *sum_input = input + col * num_rows;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, sum_input, &(output[col]), num_rows);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, sum_input, &(output[col]), num_rows);
}

cuda::CuMatrixD
cuda::test_calculate_sum_kernel(const cuda::CuMatrixD &matrix)
{
    cuda::CuMatrixD res{1, matrix.cols()};
    // each thread is one column in squared differences
    dim3 block_dim(min(1024, matrix.cols()));
    dim3 grid_dim(matrix.cols() / block_dim.x + 1);
    calculate_columnwise_sum_kernel<<<grid_dim, block_dim>>>(
        res.data(),
        matrix.data(),
        matrix.rows(),
        matrix.cols());
    CLE();
    CSC(cudaDeviceSynchronize());
    return res;
}

__global__ void
calcualte_square_difference_to_kps_database(
    double *output,
    double *query_descriptor,
    double *database_descriptors,
    int descriptor_length,
    int n_database_descriptors)
{
    // x: elements in descriptor (row), y: each descriptor (col)
    int query_row = threadIdx.x + blockDim.x * blockIdx.x;
    int query_col = 0; // query has single column here

    int database_row = query_row;
    int database_col = threadIdx.y + blockDim.y * blockIdx.y;

    int output_row = database_row, output_col = database_col;

    if (database_row >= descriptor_length || database_col >= n_database_descriptors)
        return;

    double diff =
        query_descriptor[get_index_colwise(query_row, query_col, descriptor_length)] -
        database_descriptors[get_index_colwise(database_row, database_col, descriptor_length)];

    output[get_index_colwise(output_row, output_col, descriptor_length)] = diff * diff;
}

cuda::CuMatrixD
cuda::test_calculate_difference_to_kps_database(const cuda::CuMatrixD &query, const cuda::CuMatrixD &database)
{
    assert(query.rows() == database.rows());

    cuda::CuMatrixD output(database.rows(), database.cols());

    int descriptor_length = query.rows();
    int n_database_descriptors = database.cols();

    // x: elements in descriptor (row), y: each descriptor (col)
    dim3 block_dim(min(32, descriptor_length), min(32, n_database_descriptors));
    dim3 grid_dim(descriptor_length / block_dim.x + 1, n_database_descriptors / block_dim.y + 1);
    calcualte_square_difference_to_kps_database<<<grid_dim, block_dim>>>(
        output.data(),
        query.data(),
        database.data(),
        descriptor_length,
        n_database_descriptors);
    CLE();
    CSC(cudaDeviceSynchronize());
    return output;
}

__device__ int
cal_arg_min(double *input, int num_items)
{
    cub::KeyValuePair<int, double> out;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, input, &out, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, input, &out, num_items);
    return out.key;
}

__global__ void
find_closest_keypoints_kernel(
    int *output,
    int descriptor_length,
    double *query_descriptors,
    int n_query_descriptors,
    double *database_descriptors,
    int n_database_descriptors)
{
    // this runs for each query descriptor.
    // query_idx_col is the same as query_idx
    int query_idx_col = threadIdx.x + blockDim.x * blockIdx.x;

    if (query_idx_col >= n_query_descriptors)
        return;

    double *query_descriptor = query_descriptors + query_idx_col * descriptor_length; // * sizeof(double)???

    // calculate squared differences between this query descriptors
    // and all of the database descriptors
    // squared_diffs would be matrix in shape of (n_database_descriptors X descriptor_length)
    double *squared_diffs;
    int square_differences_size = n_database_descriptors * descriptor_length * sizeof(double);
    cudaMalloc(&squared_diffs, square_differences_size);

    // x: elements in descriptor (row), y: each database descriptor (col)
    {
        dim3 block_dim(min(64, descriptor_length), min(64, n_database_descriptors));
        dim3 grid_dim(descriptor_length / block_dim.x + 1, n_database_descriptors / block_dim.y + 1);
        calcualte_square_difference_to_kps_database<<<grid_dim, block_dim>>>(
            squared_diffs,
            query_descriptor,
            database_descriptors,
            descriptor_length,
            n_database_descriptors);
    }

    // __syncthreads(); // needed??

    // calculate the distance between this query descriptor and
    // all of database descriptors, it is just columnwise sum of the differences
    // or calculate reduce sum on squared_diffs
    double *squared_dist;
    int squared_dist_bytes = sizeof(double) * n_database_descriptors;
    cudaMalloc(&squared_dist, squared_dist_bytes);
    {
        // each thread is one column in squared differences
        dim3 block_dim(min(1024, n_database_descriptors));
        dim3 grid_dim(n_database_descriptors / block_dim.x + 1);
        calculate_columnwise_sum_kernel<<<grid_dim, block_dim>>>(
            squared_dist,
            squared_diffs,
            descriptor_length,
            n_database_descriptors);
    }

    // find arg of smallest distance
    output[query_idx_col] = cal_arg_min(squared_dist, n_database_descriptors);
}

void filter_out_non_unique_matches(cuda::CuMatrixI &input)
{
    int *d_keys_in = input.data();
    thrust::device_ptr<int> indicies = thrust::device_malloc<int>(input.n_elements());
    thrust::sequence(thrust::device, indicies, indicies + input.n_elements());
    int *d_values_in = indicies.get();
    cuda::CuMatrixI keys_out(input.rows(), input.cols());
    cuda::CuMatrixI values_out(input.rows(), input.cols());
    int  *d_num_selected_out;
    cudaMalloc(&d_num_selected_out, sizeof(int));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes,
                                   d_keys_in, d_values_in,
                                   keys_out.data(), values_out.data(),
                                   d_num_selected_out, input.n_elements());
}

cuda::CuMatrixI cuda::match_descriptors(
    const cuda::CuMatrixD &query_descriptors,
    const cuda::CuMatrixD &database_descriptors,
    double match_lambda)
{
    // output length is the same as the number of keypoints in query descriptors
    // query_descriptors (desc_size X num_kp)
    // database_descriptors (desc_size X num_kp)

    // find the closest keypoint
    // one thread for each query keypoint
    // descriptors are columnwise. (like evey other CuMatrix)
    int num_query_kpts = query_descriptors.cols();
    dim3 block_dim(min(1024, num_query_kpts));
    dim3 grid_dim(num_query_kpts / block_dim.x + 1);
    cuda::CuMatrixI output(num_query_kpts, 1);
    find_closest_keypoints_kernel<<<grid_dim, block_dim>>>(
        output.data(),
        query_descriptors.rows(),
        query_descriptors.data(),
        query_descriptors.cols(),
        database_descriptors.data(),
        database_descriptors.cols());
    CLE();
    CSC(cudaDeviceSynchronize());

    // If the match is not unique, remove it.
    filter_out_non_unique_matches(output);

    return output;
}