
add_library(cuda_lib STATIC cuda_lib.cu)

add_library(using_lib STATIC using_lib.cpp)
target_link_libraries(using_lib PRIVATE cuda_lib)

add_executable(main_cuda_practice main.cpp)
target_link_libraries(main_cuda_practice using_lib)

add_executable(test_cub test_cub.cu)
