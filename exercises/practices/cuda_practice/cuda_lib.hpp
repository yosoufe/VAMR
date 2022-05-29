#pragma once
#include <stdio.h>


namespace cuda
{
void kernel_caller(size_t n_blocks, size_t n_threads);
}