set (PNP_DLT_SRC pnp_dlt.cpp pnp_dlt.hpp)

if(CMAKE_CUDA_COMPILER)
    # list(APPEND PNP_DLT_SRC .cu)
endif()

add_library(pnp_dlt ${PNP_DLT_SRC} )
get_filename_component(PNP_DLT_HEADER_DIR "./" ABSOLUTE)
target_include_directories(pnp_dlt PUBLIC ${PNP_DLT_HEADER_DIR})
target_link_libraries (pnp_dlt camera_model)