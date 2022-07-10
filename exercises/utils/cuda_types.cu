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
