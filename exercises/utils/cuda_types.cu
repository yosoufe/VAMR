#include "cuda_types.cuh"
#include "cuda_types.hpp"
#include "utils.cuh"
#include "utils.hpp"

template <typename T>
void cuda::CuMatrix<T>::free()
{
    if (d_data!= nullptr)
    {
        CSC(cudaFree(d_data));
        d_data = nullptr;
    }
}

// instantiate template struct CuMatrix
template struct cuda::CuMatrix<double>;
template struct cuda::CuMatrix<float>;


template <typename T>
cuda::CuMatrix<T> cuda::eigen_to_cuda(const MatrixT<T> &eigen)
{
    CuMatrix<T> cuda_eigen;
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
template cuda::CuMatrix<double> cuda::eigen_to_cuda<double>(const MatrixT<double> &);
template cuda::CuMatrix<float> cuda::eigen_to_cuda<float>(const MatrixT<float> &);

template <typename T>
MatrixT<T> cuda::cuda_to_eigen(const cuda::CuMatrix<T> &cuda_eigen)
{
    size_t s = cuda_eigen.n_cols * cuda_eigen.n_rows;
    T *h_data = new T[s];
    int number_of_bytes = sizeof(T) * cuda_eigen.n_rows * cuda_eigen.n_cols;
    CSC(cudaMemcpy(h_data, cuda_eigen.d_data, number_of_bytes, cudaMemcpyDeviceToHost));
    MatrixT<T> res;
    res = MatrixT<T>::Map(h_data, cuda_eigen.n_rows, cuda_eigen.n_cols);
    return res;
}

// instantiate template function above
template MatrixT<double> cuda::cuda_to_eigen(const cuda::CuMatrix<double> &cuda_eigen);
template MatrixT<float> cuda::cuda_to_eigen(const cuda::CuMatrix<float> &cuda_eigen);