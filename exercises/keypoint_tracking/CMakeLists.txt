set (TARGET_NAME keypoint_tracking)
set (SRC keypoint_tracking.cpp)

if(CUDA_AVAILABLE)
    list(APPEND SRC keypoint_tracking.cu)
endif()


add_library(${TARGET_NAME} ${SRC})
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (${TARGET_NAME} utils ${THRSUT_LIB_FROM_SOURCE})
