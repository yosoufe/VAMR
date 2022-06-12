set (TARGET_NAME utils)

set(SRC utils.cpp folder_manager.cpp)

if(CMAKE_CUDA_COMPILER)
    list(APPEND SRC utils.cu)
endif()

add_library(${TARGET_NAME} STATIC ${SRC})
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME} Eigen3::Eigen ${OpenCV_LIBS} ${OpenMP_LIBRARIES} tbb)
