#include "cuda_types.cuh"
#include "cuda_types.hpp"
#include "utils.cuh"
#include "utils.hpp"

template <typename T>
void cuda::CuMatrixDeleter<T>::operator() (T * p) const
{
    if (p != nullptr)
    {
        std::cout << "freeing the gpu memory" << std::endl;
        CSC(cudaFree(p));
        std::cout << "gpu memory freed" << std::endl;
    }
}

template <typename T>
cuda::CuMatrix<T> cuda::CuMatrix<T>::factory(T* ptr, int n_cols, int n_rows)
{
    cuda::CuMatrix<T> res;
    res.d_data = std::shared_ptr<T>(ptr, cuda::CuMatrixDeleter<T>());
    res.n_cols = n_cols;
    res.n_rows = n_rows;
    return res;
}

// instantiate template struct CuMatrix
template struct cuda::CuMatrixDeleter<double>;
template struct cuda::CuMatrixDeleter<float>;
template struct cuda::CuMatrix<double>;
template struct cuda::CuMatrix<float>;


template <typename T>
cuda::CuMatrix<T> cuda::eigen_to_cuda(const MatrixT<T> &eigen)
{
    int number_of_bytes = sizeof(T) * eigen.cols() * eigen.rows();
    T* output_ptr;
    CSC(cudaMalloc(&output_ptr, number_of_bytes));
    CSC(cudaMemcpy(output_ptr, eigen.data(), number_of_bytes, cudaMemcpyHostToDevice));
    // print_cuda_eigen<T><<<1, 1>>>(cuda_eigen.d_data, eigen.cols(), eigen.rows());
    cudaDeviceSynchronize();
    return cuda::CuMatrix<T>::factory(output_ptr, eigen.cols(), eigen.rows());
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

