#include "camera_model.hpp"
#include "utils.cuh"

__global__ void undistort_image_kernel(
    unsigned char *input, unsigned char *output,
    double u0, double v0,
    double d1, double d2)
{
    auto row = blockIdx.x;
    auto n_rows = gridDim.x;
    auto col = threadIdx.x;
    auto n_cols = blockDim.x;
    auto u = col;
    auto v = row;
    double r_2 = (u - u0) * (u - u0) + (v - v0) * (v - v0);
    double c = 1 + d1 * r_2 + d2 * r_2 * r_2;
    int u_d = c * (u - u0) + u0;
    int v_d = c * (v - v0) + v0;

    auto undistorted_idx = get_index_rowwise(v, u, n_cols, 3);

    if (u_d >= 0 && u_d < n_cols && v_d >= 0 && v_d < n_rows)
    {
        auto distorted_idx = get_index_rowwise(v_d, u_d, n_cols, 3);
        output[undistorted_idx] = input[distorted_idx];
        output[undistorted_idx + 1] = input[distorted_idx + 1];
        output[undistorted_idx + 2] = input[distorted_idx + 2];
    }
    else
    {
        output[undistorted_idx] = 0;
        output[undistorted_idx + 1] = 0;
        output[undistorted_idx + 2] = 0;
    }
}

cv::Mat cuda::undistort_image(const cv::Mat &src_img,
                              double d1,
                              double d2,
                              const Eigen::Vector2d &principal_pt)
{
    if (src_img.type() != CV_8UC3)
        throw std::runtime_error("Type of input image should be CV_8UC3");
    cv::Mat res = src_img.clone();
    double u0 = principal_pt(0);
    double v0 = principal_pt(1);

    unsigned char *d_input, *d_output;
    const int number_of_bytes = src_img.step * src_img.rows;

    // Allocate device memory
    CSC(cudaMalloc(&d_input, number_of_bytes));
    CSC(cudaMalloc(&d_output, number_of_bytes));

    CSC(cudaMemcpy(d_input, src_img.data, number_of_bytes, cudaMemcpyHostToDevice));

    undistort_image_kernel<<<src_img.rows, src_img.cols>>>(d_input, d_output, u0, v0, d1, d2);
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(res.data, d_output, number_of_bytes, cudaMemcpyDeviceToHost));
    CSC(cudaFree(d_input));
    CSC(cudaFree(d_output));
    return res;
}