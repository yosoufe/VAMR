#include "operations.cuh"
#include "operations.hpp"
#include "utils.cuh"

cuda::CuMatrixD cuda::sobel_x_kernel()
{
    Eigen::MatrixXd kernel = ::sobel_x_kernel();
    return cuda::eigen_to_cuda(kernel);
}

cuda::CuMatrixD cuda::sobel_y_kernel()
{
    Eigen::MatrixXd kernel = ::sobel_y_kernel();
    return cuda::eigen_to_cuda(kernel);
}

cuda::CuMatrixD cuda::correlation(const cuda::CuMatrixD &input, const cuda::CuMatrixD &kernel)
{
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    // input
    const int in_n = 1;
    const int in_c = 1;
    const int in_h = input.n_rows;
    const int in_w = input.n_cols;

    cudnnTensorDescriptor_t in_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
        in_n, in_c, in_h, in_w));

    double *in_data = input.d_data;

    // filter
    const int filt_k = 1;
    const int filt_c = 1;
    const int filt_h = kernel.n_rows;
    const int filt_w = kernel.n_cols;

    cudnnFilterDescriptor_t filt_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

    double *filt_data = kernel.d_data;

    // convolution
    const int pad_h = 1;
    const int pad_w = 1;
    const int str_h = 1;
    const int str_w = 1;
    const int dil_h = 1;
    const int dil_w = 1;

    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

    // output
    int out_n;
    int out_c;
    int out_h;
    int out_w;

    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

    cudnnTensorDescriptor_t out_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
        out_n, out_c, out_h, out_w));

    double *out_data;
    CSC(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(double)));

    // algorithm
    cudnnConvolutionFwdAlgoPerf_t algo;
    int perf_count;
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        in_desc, filt_desc, conv_desc, out_desc,
        1, &perf_count, &algo));

    // workspace
    size_t ws_size;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo.algo, &ws_size));

    double *ws_data;
    CSC(cudaMalloc(&ws_data, ws_size));

    // perform
    double alpha = -1.0;
    double beta = 0;

    CUDNN_CALL(cudnnConvolutionForward(
        cudnn,
        &alpha, in_desc, in_data, filt_desc, filt_data,
        conv_desc, algo.algo, ws_data, ws_size,
        &beta, out_desc, out_data));

    cuda::CuMatrixD out;
    out.d_data = out_data;
    out.n_cols = out_w;
    out.n_rows = out_h;

    // finalizing
    CSC(cudaFree(ws_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CALL(cudnnDestroy(cudnn));

    return out;
}