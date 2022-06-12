#include "utils.cuh"
#include "utils.hpp"

template <typename T>
void cuda::EigenCuda<T>::free()
{
    CSC(cudaFree(d_data));
}

// instantiate template struct EigenCuda
template struct cuda::EigenCuda<double>;
template struct cuda::EigenCuda<float>;

__device__ int
get_index_rowwise(int row, int col, int n_cols, int stride)
{
    // printf("row: %d, col: %d, n_cols: %d\n", row, col, n_cols);
    return (col + n_cols * row) * stride;
}

__device__ int
get_index_colwise(int row, int col, int n_rows, int stride)
{
    // printf("row: %d, col: %d, n_rows: %d\n", row, col, n_rows);
    return (row + n_rows * col) * stride;
}

template <typename T>
__global__ void print_cuda_eigen(T *data, int cols, int rows)
{
    printf("printing in cuda kernel:\n");
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            int idx = get_index_colwise(row, col, rows, 1);
            printf("%d: ", idx);
            printf("%f ,", float(data[idx]));
        }
        printf("\n");
    }
}

template <typename T>
cuda::EigenCuda<T> cuda::eigen_to_cuda(const MatrixT<T> &eigen)
{
    EigenCuda<T> cuda_eigen;
    int number_of_bytes = sizeof(T) * eigen.cols() * eigen.rows();
    CSC(cudaMalloc(&cuda_eigen.d_data, number_of_bytes));
    CSC(cudaMemcpy(cuda_eigen.d_data, eigen.data(), number_of_bytes, cudaMemcpyHostToDevice));
    // print_cuda_eigen<T><<<1, 1>>>(cuda_eigen.d_data, eigen.cols(), eigen.rows());
    cudaDeviceSynchronize();
    cuda_eigen.n_cols = eigen.cols();
    cuda_eigen.n_rows = eigen.rows();
    return cuda_eigen;
}

// instantiate template function above
template cuda::EigenCuda<double> cuda::eigen_to_cuda<double>(const MatrixT<double> &);
template cuda::EigenCuda<float> cuda::eigen_to_cuda<float>(const MatrixT<float> &);

template <typename T>
cuda::MatrixT<T> cuda::cuda_to_eigen(const cuda::EigenCuda<T> &cuda_eigen)
{
    size_t s = cuda_eigen.n_cols * cuda_eigen.n_rows;
    T *h_data = new T[s];
    int number_of_bytes = sizeof(T) * cuda_eigen.n_rows * cuda_eigen.n_cols;
    CSC(cudaMemcpy(h_data, cuda_eigen.d_data, number_of_bytes, cudaMemcpyDeviceToHost));
    cuda::MatrixT<T> res;
    res = cuda::MatrixT<T>::Map(h_data, cuda_eigen.n_rows, cuda_eigen.n_cols);
    return res;
}

// instantiate template function above
template cuda::MatrixT<double> cuda::cuda_to_eigen(const cuda::EigenCuda<double> &cuda_eigen);
template cuda::MatrixT<float> cuda::cuda_to_eigen(const cuda::EigenCuda<float> &cuda_eigen);