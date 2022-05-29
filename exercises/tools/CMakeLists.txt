
set(UTILS_SRC 
    utils.cpp
    folder_manager.cpp
)

if(CMAKE_CUDA_COMPILER)
    list(APPEND UTILS_SRC utils.cu)
endif()

add_library(utils SHARED
    ${UTILS_SRC}
)
target_link_libraries(utils Eigen3::Eigen ${OpenCV_LIBS})
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})