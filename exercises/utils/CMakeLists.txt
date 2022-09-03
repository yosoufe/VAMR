set (TARGET_NAME utils)

set(SRC utils.cpp folder_manager.cpp operations.cpp cuda_types.cpp)

if(CUDA_AVAILABLE)
    list(APPEND SRC utils.cu operations.cu cuda_types.cu)
endif()

add_library(${TARGET_NAME} STATIC ${SRC})
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    ${TARGET_NAME}
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${OpenMP_LIBRARIES}
    tbb 
    cudnn
)
