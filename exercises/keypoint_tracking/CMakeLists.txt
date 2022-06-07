set (TARGET_NAME keypoint_tracking)
set (SRC keypoint_tracking.cpp)

if(CMAKE_CUDA_COMPILER)
    # list(APPEND SRC )
endif()


add_library(${TARGET_NAME} ${SRC})
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (${TARGET_NAME} utils)