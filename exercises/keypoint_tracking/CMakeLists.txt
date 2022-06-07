
set (KEYPOINT_TRACKING_MODEL_SRC )

if(CMAKE_CUDA_COMPILER)
    # list(APPEND KEYPOINT_TRACKING_MODEL_SRC )
endif()


add_library(keypoint_tracking ${KEYPOINT_TRACKING_MODEL_SRC} )
get_filename_component(KEYPOINT_TRACKING_HEADER_DIR "./" ABSOLUTE)
target_include_directories(keypoint_tracking PUBLIC CMAKE_SOURCE_DIR)
target_link_libraries (keypoint_tracking utils)