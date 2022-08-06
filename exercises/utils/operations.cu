#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>

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

void setTensorDesc(cudnnTensorDescriptor_t &tensorDesc,
                   const cudnnTensorFormat_t &tensorFormat,
                   const cudnnDataType_t &dataType,
                   int n,
                   int c,
                   int h,
                   int w)
{
#define ND_TENSOR_DESCRIPTOR
#if SIMPLE_TENSOR_DESCRIPTOR
    CUDNN_CALL(cudnnSetTensor4dDescriptor(tensorDesc,
                                          tensorFormat,
                                          dataType,
                                          n, c,
                                          h,
                                          w));
#elif defined(ND_TENSOR_DESCRIPTOR)
    const int nDims = 4;
    int dimA[nDims] = {n, c, h, w};
    int strideA[nDims] = {c * h * w, h * w, w, 1};
    CUDNN_CALL(cudnnSetTensorNdDescriptor(tensorDesc,
                                          dataType,
                                          4,
                                          dimA,
                                          strideA));
#else
    CUDNN_CALL(cudnnSetTensor4dDescriptorEx(tensorDesc,
                                            dataType,
                                            n, c,
                                            h, w,
                                            c * h * w, h * w, w, 1));
#endif
}

cuda::CuMatrixD cuda::correlation(const cuda::CuMatrixD &input, const cuda::CuMatrixD &kernel)
{
    /**
     * in CUDNN it seems it is
     * height <-> number of columns
     * weight <-> number of rows.
     */
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));
    CUDNN_CALL(cudnnCnnInferVersionCheck());

    // input
    const int in_n = 1;
    const int in_c = 1;
    const int in_h = input.cols();
    const int in_w = input.rows();

    cudnnTensorDescriptor_t in_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
    // CUDNN_CALL(cudnnSetTensor4dDescriptor(
    //     in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
    //     in_n, in_c, in_h, in_w));

    setTensorDesc(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
                  in_n, in_c, in_h, in_w);

    double *in_data = input.data();

    // filter
    const int filt_k = 1;
    const int filt_c = 1;
    const int filt_h = kernel.cols();
    const int filt_w = kernel.rows();

    cudnnFilterDescriptor_t filt_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

    double *filt_data = kernel.data();

    // convolution
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));

    const int convDims = 2;
    int padA[convDims] = {kernel.cols() / 2, kernel.rows() / 2};
    int filterStrideA[convDims] = {1, 1};
    int upscaleA[convDims] = {1, 1};
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(
        conv_desc, convDims, padA, filterStrideA,
        upscaleA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));

    // output
    int out_n;
    int out_c;
    int out_h;
    int out_w;

    const int tensorDims = 4;
    int tensorOuputDimA[tensorDims];
    CUDNN_CALL(cudnnGetConvolutionNdForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        tensorDims, tensorOuputDimA));

    out_n = tensorOuputDimA[0];
    out_c = tensorOuputDimA[1];
    out_h = tensorOuputDimA[2];
    out_w = tensorOuputDimA[3];

    // std::cout << " out_n " << out_n << " out_c " << out_c;
    // std::cout << " out_h " << out_h << " out_w " << out_w << std::endl;

    cudnnTensorDescriptor_t out_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));

    setTensorDesc(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
                  out_n, out_c, out_h, out_w);

    double *out_data;
    CSC(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(double)));

    // algorithm
    cudnnConvolutionFwdAlgo_t algo;
    int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgoCount = -1;
    cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
        cudnn,
        in_desc, filt_desc, conv_desc, out_desc,
        requestedAlgoCount, &returnedAlgoCount, results));
    algo = results[0].algo;

    // workspace
    size_t ws_size;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

    double *ws_data;
    if (ws_size != 0)
        CSC(cudaMalloc(&ws_data, ws_size));

    // perform
    double alpha = 1.0;
    double beta = 0;

    CUDNN_CALL(cudnnConvolutionForward(
        cudnn,
        &alpha, in_desc, in_data, filt_desc, filt_data,
        conv_desc, algo, ws_data, ws_size,
        &beta, out_desc, out_data));

    // zeros out the elements that are calculated by the padding
    auto out = cuda::CuMatrixD(out_data, out_w, out_h);
    int s_row = kernel.rows() / 2;
    int s_col = kernel.cols() / 2;
    int l_row = input.rows() - 2 * s_row;
    int l_col = input.cols() - 2 * s_col;

    zero_borders(out, s_row, s_col, l_row, l_col);

    // finalizing
    if (ws_size != 0)
        CSC(cudaFree(ws_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CALL(cudnnDestroy(cudnn));

    return out;
}

template <typename functor>
void _unary_operator(const cuda::CuMatrixD &input, cuda::CuMatrixD &output, functor unary_f)
{
    auto d_vec_start = cuda::thrust_ptr_begin(input);
    auto d_vec_end = cuda::thrust_ptr_end(input);
    auto d_output_start = cuda::thrust_ptr_begin(output);
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
    cuda::CuMatrixD res(input.rows(), input.cols());
    _unary_operator(input, res, power<double>(pow));
    return res;
}

cuda::CuMatrixD cuda::pow(cuda::CuMatrixD &&input, double pow)
{
    _unary_operator(input, input, power<double>(pow));
    return input;
}

double cuda::norm(const cuda::CuMatrixD &input)
{
    auto squared = cuda::pow(input, 2);
    auto sum_of_squared = thrust::reduce(
        thrust::cuda::par,
        cuda::thrust_ptr_begin(squared), 
        cuda::thrust_ptr_end(squared));
    return std::sqrt(sum_of_squared);
}

template <typename functor>
void _binary_operator(const cuda::CuMatrixD &i1, const cuda::CuMatrixD &i2, cuda::CuMatrixD &output, functor binary_f)
{
    auto s1 = cuda::thrust_ptr_begin(i1);
    auto e1 = cuda::thrust_ptr_end(i1);
    auto s2 = cuda::thrust_ptr_begin(i2);
    auto output_ptr = cuda::thrust_ptr_begin(output);
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
    cuda::CuMatrixD out(i1.rows(), i1.cols());
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
    cuda::CuMatrixD out(i1.rows(), i1.cols());
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
    cuda::CuMatrixD out(i1.rows(), i1.cols());
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
    cuda::CuMatrixD out(mat.rows(), mat.cols());
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
    cuda::CuMatrixD out(input.rows(), input.cols());
    _unary_operator(input, out, thrshold_lower_functor<double>(threshold, substitute));
    return out;
}

cuda::CuMatrixD cuda::threshold_lower(cuda::CuMatrixD &&input, double threshold, double substitute)
{
    _unary_operator(input, input, thrshold_lower_functor<double>(threshold, substitute));
    return input;
}

template <typename T>
struct ZeroBorderOperator
{
    int n_rows, n_cols;
    int s_row, s_col, l_row, l_col;
    ZeroBorderOperator(int n_rows, int n_cols,
                       int s_row, int s_col,
                       int l_row, int l_col) : n_rows(n_rows),
                                               n_cols(n_cols),
                                               s_row(s_row),
                                               s_col(s_col),
                                               l_row(l_row),
                                               l_col(l_col)
    {
    }

    template <typename Tuple>
    __host__ __device__ T operator()(Tuple t)
    {
        int index = thrust::get<1>(t);
        T value = thrust::get<0>(t);
        auto idx = get_2d_index_colwise(index, n_rows);
        auto row = thrust::get<0>(idx);
        auto col = thrust::get<1>(idx);

        if (col < s_col || col >= s_col + l_col || row < s_row || row >= s_row + l_row)
            return T(0.0);
        else
            return value;
    }
};

thrust::device_ptr<int> cuda::create_indices(const cuda::CuMatrixD &input)
{
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(input.n_elements());
    thrust::sequence(thrust::device, d_output, d_output + input.n_elements());
    return d_output;
}

void cuda::zero_borders(cuda::CuMatrixD &input, int s_row, int s_col, int l_row, int l_col)
{
    auto d_indices_start = cuda::create_indices(input);
    auto &output = input;
    thrust::device_ptr<double> d_vec_start = thrust::device_pointer_cast(input.data());
    thrust::device_ptr<double> d_vec_end = d_vec_start + input.cols() * input.rows();
    thrust::device_ptr<double> d_output_start = thrust::device_pointer_cast(output.data());
    ZeroBorderOperator<double> ops(input.rows(), input.cols(), s_row, s_col, l_row, l_col);
    thrust::transform(thrust::cuda::par,
                      thrust::make_zip_iterator(thrust::make_tuple(d_vec_start, d_indices_start)),
                      thrust::make_zip_iterator(thrust::make_tuple(d_vec_end, d_indices_start + input.n_elements())),
                      d_output_start, ops);
}