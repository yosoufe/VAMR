
set (CAMERA_MODEL_SRC camera_model.cpp camera_model.hpp)

if(CMAKE_CUDA_COMPILER)
    list(APPEND CAMERA_MODEL_SRC camera_model.cu)
endif()


add_library(camera_model ${CAMERA_MODEL_SRC} )
get_filename_component(CAMERA_MODEL_HEADER_DIR "./" ABSOLUTE)
target_include_directories(camera_model PUBLIC ${CAMERA_MODEL_HEADER_DIR})
target_link_libraries (camera_model utils)