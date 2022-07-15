#include "cuda_types.cuh"
#include "cuda_types.hpp"
#include "utils.cuh"
#include "utils.hpp"

template <typename T>
void cuda::CuMatrixDeleter<T>::operator()(T *p) const
{
    if (p != nullptr)
    {
        // FIXME: this should not throw exception.
        // or exit. CSC might exit.
        // std::cout << "freeing gpu memory" << std::endl;
        CSC(cudaFree(p));
        p = nullptr;
    }
}

template <typename T>
cuda::CuMatrix<T>::CuMatrix() : cuda::CuMatrix<T>(nullptr, 0, 0)
{
}

template <typename T>
cuda::CuMatrix<T>::CuMatrix(T *ptr, int n_cols, int n_rows) : d_data(
                                                                  std::shared_ptr<T>(ptr,
                                                                                     cuda::CuMatrixDeleter<T>())),
                                                              n_cols(n_cols),
                                                              n_rows(n_rows)
{
}

template <typename T>
cuda::CuMatrix<T>::CuMatrix(int cols, int rows) : n_cols(cols), n_rows(rows)
{
    int number_of_bytes = sizeof(T) * cols * rows;
    T *ptr;
    CSC(cudaMalloc(&ptr, number_of_bytes));
    d_data = std::shared_ptr<T>(ptr,
                                cuda::CuMatrixDeleter<T>());
}

template <typename T>
cuda::CuMatrix<T> cuda::CuMatrix<T>::clone() const
{
    cuda::CuMatrix<T> output(n_cols, n_rows);
    CSC(cudaMemcpy(output.d_data.get(), d_data.get(), sizeof(T) * n_elements(), cudaMemcpyDeviceToDevice));
    return output;
}

template <typename T>
__global__ void
copy_block_kernel(T *src, T *dst,
                  int src_n_rows,
                  int src_row_offset, int src_col_offset,
                  int dst_n_rows, int dst_n_cols)
{
    int dst_row = threadIdx.x + blockDim.x * blockIdx.x;
    int dst_col = threadIdx.y + blockDim.y * blockIdx.y;

    if (dst_row >= dst_n_rows || dst_col >= dst_n_cols)
        return;

    int src_row = dst_row + src_row_offset;
    int src_col = dst_col + src_col_offset;
    dst[get_index_colwise(dst_row, dst_col, dst_n_rows)] =
        src[get_index_colwise(src_row, src_col, src_n_rows)];
}

template <typename T>
cuda::CuMatrix<T> cuda::CuMatrix<T>::block(int row, int col, int height, int width) const
{
    cuda::CuMatrix<T> output(width, height);

    // super slow for large matrices.
    auto using_stream_impl = [&]()
    {
        cudaStream_t streams[width];
        int counter = 0;
        for (int current_col = col; current_col < col + width; ++current_col, ++counter)
        {
            T *src = d_data.get() +
                     get_index_colwise(row, current_col, n_rows);
            T *dst = output.d_data.get() +
                     get_index_colwise(0, counter, height);
            CSC(cudaStreamCreate(&streams[counter]));
            CSC(cudaMemcpyAsync(dst, src, height * sizeof(T), cudaMemcpyDeviceToDevice, streams[counter]));
            cudaDeviceSynchronize();
        }
    };

    auto using_kernel_impl = [&]()
    {
        dim3 block_dim;
        block_dim.x = min(32, height);
        block_dim.y = min(32, width);

        dim3 grid_dim;
        grid_dim.x = height / block_dim.x + 1;
        grid_dim.y = width / block_dim.y + 1;

        copy_block_kernel<T><<<grid_dim, block_dim>>>(
            d_data.get(), output.d_data.get(),
            n_rows,
            row, col,
            height, width);
    };

    using_stream_impl();
    // using_kernel_impl();

    return output;
}

template struct cuda::CuMatrixDeleter<double>;
template struct cuda::CuMatrixDeleter<float>;
template struct cuda::CuMatrix<double>;
template struct cuda::CuMatrix<float>;

template <typename T>
cuda::CuMatrix<T> cuda::eigen_to_cuda(const MatrixT<T> &eigen)
{
    int number_of_bytes = sizeof(T) * eigen.cols() * eigen.rows();
    T *output_ptr;
    CSC(cudaMalloc(&output_ptr, number_of_bytes));
    CSC(cudaMemcpy(output_ptr, eigen.data(), number_of_bytes, cudaMemcpyHostToDevice));
    // print_cuda_eigen<T><<<1, 1>>>(cuda_eigen.d_data.get(), eigen.cols(), eigen.rows());
    cudaDeviceSynchronize();
    return cuda::CuMatrix<T>(output_ptr, eigen.cols(), eigen.rows());
}

// instantiate template function above
template cuda::CuMatrix<double> cuda::eigen_to_cuda<double>(const MatrixT<double> &);
template cuda::CuMatrix<float> cuda::eigen_to_cuda<float>(const MatrixT<float> &);

template <typename T>
MatrixT<T> cuda::cuda_to_eigen(const cuda::CuMatrix<T> &cuda_eigen)
{
    size_t s = cuda_eigen.n_cols * cuda_eigen.n_rows;
    T *h_data = new T[s];
    int number_of_bytes = sizeof(T) * s;
    CSC(cudaMemcpy(h_data, cuda_eigen.d_data.get(), number_of_bytes, cudaMemcpyDeviceToHost));
    MatrixT<T> res;
    res = MatrixT<T>::Map(h_data, cuda_eigen.n_rows, cuda_eigen.n_cols);
    return res;
}

// instantiate template function above
template MatrixT<double> cuda::cuda_to_eigen(const cuda::CuMatrix<double> &cuda_eigen);
template MatrixT<float> cuda::cuda_to_eigen(const cuda::CuMatrix<float> &cuda_eigen);

bool are_matrices_close(const cuda::CuMatrixD &first, const Eigen::MatrixXd &second)
{
    auto host_matrix = cuda::cuda_to_eigen(first);
    return are_matrices_close(host_matrix, second);
}