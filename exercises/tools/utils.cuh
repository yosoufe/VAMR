#pragma once 
#include <iostream>
#include <cuda_runtime.h>

/**
 * @brief      CUDA safe call.
 *
 * @param[in]  err          The error
 * @param[in]  msg          The message
 * @param[in]  file_name    The file name
 * @param[in]  line_number  The line number
 */
inline void _safe_cuda_call(cudaError err, const char* file_name, const int line_number)
{
    if(err!=cudaSuccess) {
        fprintf(stderr,"\nFile: %s:%d\nReason: %s\n\n",file_name,line_number,cudaGetErrorString(err));
        std::cin.get();
        exit(EXIT_FAILURE);
    }
}

inline void _cuda_last_error(const char* file_name, const int line_number)
{
    auto cuda_erro = cudaGetLastError();
    _safe_cuda_call(cuda_erro, file_name, line_number);
}

// Safe call
#define CSC(call) _safe_cuda_call((call),__FILE__,__LINE__)
// CUDA LAST Error
#define CLE() _cuda_last_error(__FILE__,__LINE__)