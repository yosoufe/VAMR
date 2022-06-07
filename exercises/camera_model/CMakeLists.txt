set (TARGET_NAME camera_model)
set (SRC camera_model.cpp camera_model.hpp)

if(CMAKE_CUDA_COMPILER)
    list(APPEND SRC camera_model.cu)
endif()


add_library(${TARGET_NAME} ${SRC} )
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (${TARGET_NAME} utils)