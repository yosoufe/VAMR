#include "using_lib.hpp"
#include "cuda_lib.hpp"

void use_the_lib(){
    cuda::kernel_caller(20,10);
}