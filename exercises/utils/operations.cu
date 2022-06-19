#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/transform.h>

#include "operations.cuh"
#include "operations.hpp"
#include "utils.cuh"
#include <Eigen/Dense>

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

cuda::CuMatrixD cuda::ones(int rows, int cols)
{
    Eigen::MatrixXd m = Eigen::MatrixXd::Ones(rows, cols);
    return cuda::eigen_to_cuda(m);
}

cuda::CuMatrixD cuda::correlation(const cuda::CuMatrixD &input, const cuda::CuMatrixD &kernel)
{
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));
    CUDNN_CALL(cudnnCnnInferVersionCheck());

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

    double *in_data = input.d_data.get();

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

    double *filt_data = kernel.d_data.get();

    // convolution
    const int pad_h = kernel.n_cols / 2;
    const int pad_w = kernel.n_rows / 2;
    const int str_h = 1;
    const int str_w = 1;
    const int dil_h = 1;
    const int dil_w = 1;

    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));

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
    double alpha = 1.0;
    double beta = 0;

    CUDNN_CALL(cudnnConvolutionForward(
        cudnn,
        &alpha, in_desc, in_data, filt_desc, filt_data,
        conv_desc, algo.algo, ws_data, ws_size,
        &beta, out_desc, out_data));

    auto out = cuda::CuMatrixD(out_data, out_w, out_h);

    // finalizing
    CSC(cudaFree(ws_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CALL(cudnnDestroy(cudnn));

    return out;
}

template <typename functor>
void _unary_operator(const cuda::CuMatrixD &input,cuda::CuMatrixD &output, functor unary_f)
{
    thrust::device_ptr<double> d_vec_start = thrust::device_pointer_cast(input.d_data.get());
    thrust::device_ptr<double> d_vec_end = d_vec_start + input.n_cols * input.n_rows;
    thrust::device_ptr<double> d_output_start = thrust::device_pointer_cast(output.d_data.get());
    thrust::transform(thrust::cuda::par, d_vec_start, d_vec_end, d_output_start, unary_f);
}


template <typename T>
struct power
{
    T p;
    power(T p) : p(p){};
    __host__ __device__ T operator()(const T &x) const
    {
        return pow(x, p);
    }
};

cuda::CuMatrixD cuda::pow(const cuda::CuMatrixD &input, double pow)
{
    cuda::CuMatrixD res(input.n_cols, input.n_rows);
    _unary_operator(input, res, power<double>(pow));
    return res;
}


cuda::CuMatrixD cuda::pow(cuda::CuMatrixD &&input, double pow)
{
    _unary_operator(input, input, power<double>(pow));
    return input;
}

template <typename functor>
void _binary_operator(const cuda::CuMatrixD &i1, const cuda::CuMatrixD &i2, cuda::CuMatrixD &output, functor binary_f)
{
    thrust::device_ptr<double> s1 = thrust::device_pointer_cast(i1.d_data.get());
    thrust::device_ptr<double> e1 = s1 + i1.n_cols * i1.n_rows;
    thrust::device_ptr<double> s2 = thrust::device_pointer_cast(i2.d_data.get());
    thrust::device_ptr<double> output_ptr = thrust::device_pointer_cast(output.d_data.get());
    thrust::transform(thrust::cuda::par, s1, e1, s2, output_ptr, binary_f);
}

template <typename T>
struct multiply_functor
    : public thrust::binary_function<T, T, T>
{
    __host__ __device__ T operator()(T x, T y)
    {
        return x * y;
    }
};

cuda::CuMatrixD cuda::operator*(const cuda::CuMatrixD &i1, const cuda::CuMatrixD &i2)
{
    cuda::CuMatrixD out(i1.n_cols, i1.n_rows);
    _binary_operator(i1, i2, out, multiply_functor<double>());
    return out;
}

cuda::CuMatrixD cuda::operator*(cuda::CuMatrixD &&i1, const cuda::CuMatrixD &i2)
{
    _binary_operator(i1, i2, i1, multiply_functor<double>());
    return i1;
}

cuda::CuMatrixD cuda::operator*(const cuda::CuMatrixD &i1, cuda::CuMatrixD &&i2)
{
    _binary_operator(i1, i2, i2, multiply_functor<double>());
    return i2;
}

cuda::CuMatrixD cuda::operator*(cuda::CuMatrixD &&i1, cuda::CuMatrixD &&i2)
{
    _binary_operator(i1, i2, i1, multiply_functor<double>());
    return i1;
}

template <typename T>
struct plus_functor
    : public thrust::binary_function<T, T, T>
{
    __host__ __device__ T operator()(T x, T y)
    {
        return x + y;
    }
};

cuda::CuMatrixD cuda::operator+(const cuda::CuMatrixD &i1, const cuda::CuMatrixD &i2)
{
    cuda::CuMatrixD out(i1.n_cols, i1.n_rows);
    _binary_operator(i1, i2, out, plus_functor<double>());
    return out;
}

cuda::CuMatrixD cuda::operator+(cuda::CuMatrixD &&i1, const cuda::CuMatrixD &i2)
{
    _binary_operator(i1, i2, i1, plus_functor<double>());
    return i1;
}

cuda::CuMatrixD cuda::operator+(const cuda::CuMatrixD &i1, cuda::CuMatrixD &&i2)
{
    _binary_operator(i1, i2, i2, plus_functor<double>());
    return i2;
}

cuda::CuMatrixD cuda::operator+(cuda::CuMatrixD &&i1, cuda::CuMatrixD &&i2)
{
    _binary_operator(i1, i2, i1, plus_functor<double>());
    return i1;
}

template <typename T>
struct minus_functor
    : public thrust::binary_function<T, T, T>
{
    __host__ __device__ T operator()(T x, T y)
    {
        return x - y;
    }
};

cuda::CuMatrixD cuda::operator-(const cuda::CuMatrixD &i1, const cuda::CuMatrixD &i2)
{
    cuda::CuMatrixD out(i1.n_cols, i1.n_rows);
    _binary_operator(i1, i2, out, minus_functor<double>());
    return out;
}

cuda::CuMatrixD cuda::operator-(cuda::CuMatrixD &&i1, const cuda::CuMatrixD &i2)
{
    _binary_operator(i1, i2, i1, minus_functor<double>());
    return i1;
}

cuda::CuMatrixD cuda::operator-(const cuda::CuMatrixD &i1, cuda::CuMatrixD &&i2)
{
    _binary_operator(i1, i2, i2, minus_functor<double>());
    return i2;
}

cuda::CuMatrixD cuda::operator-(cuda::CuMatrixD &&i1, cuda::CuMatrixD &&i2)
{
    _binary_operator(i1, i2, i1, minus_functor<double>());
    return i1;
}

template <typename T>
struct multiply_by_constant
{
    T cst;
    multiply_by_constant(T cst) : cst(cst){};
    __host__ __device__ T operator()(const T &x) const
    {
        return x * cst;
    }
};


cuda::CuMatrixD cuda::operator*(const cuda::CuMatrixD &mat, double constant)
{
    cuda::CuMatrixD out(mat.n_cols, mat.n_rows);
    _unary_operator(mat, out, multiply_by_constant<double>(constant));
    return out;
}

cuda::CuMatrixD cuda::operator*(double constant, const cuda::CuMatrixD &mat)
{
    return cuda::operator*(mat, constant);
}

cuda::CuMatrixD cuda::operator*(cuda::CuMatrixD &&mat, double constant)
{
    _unary_operator(mat, mat, multiply_by_constant<double>(constant));
    return mat;
}

cuda::CuMatrixD cuda::operator*(double constant, cuda::CuMatrixD &&mat)
{
    _unary_operator(mat, mat, multiply_by_constant<double>(constant));
    return mat;
}

template <typename T>
struct thrshold_lower_functor
{
    T threshold;
    T substitute;
    thrshold_lower_functor(T th, T sub) : threshold(th), substitute(sub){};
    __host__ __device__ T operator()(const T &x) const
    {
        if (x < threshold)
            return substitute;
        return x;
    }
};


cuda::CuMatrixD cuda::threshold_lower(const cuda::CuMatrixD &input, double threshold, double substitute)
{
    cuda::CuMatrixD out(input.n_cols, input.n_rows);
    _unary_operator(input, out, thrshold_lower_functor<double>(threshold, substitute));
    return out;
}

cuda::CuMatrixD cuda::threshold_lower(cuda::CuMatrixD &&input, double threshold, double substitute)
{
    _unary_operator(input, input, thrshold_lower_functor<double>(threshold, substitute));
    return input;
}