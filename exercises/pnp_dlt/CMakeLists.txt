set (TARGET_NAME pnp_dlt)
set (SRC pnp_dlt.cpp pnp_dlt.hpp)

if(CMAKE_CUDA_COMPILER)
    # list(APPEND SRC .cu)
endif()

add_library(${TARGET_NAME} ${SRC} )
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (${TARGET_NAME} camera_model)