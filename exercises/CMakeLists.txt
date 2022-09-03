cmake_minimum_required(VERSION 3.0)
project(VAMR LANGUAGES CXX)


include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA found")
    set(CUDA_AVAILABLE True)
    enable_language(CUDA)
    add_compile_definitions(WITH_CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CUDA_STANDARD 14)
    # example to use cuda from source with cmake:
    # https://github.com/NVIDIA/thrust/blob/main/examples/cmake/add_subdir/CMakeLists.txt
    find_package(Thrust 2.1.0.0 REQUIRED CONFIG
        NO_DEFAULT_PATH
        HINTS
            "/3rdparty/thrust/thrust/cmake"
        )
    
    set(THRUST_REQUIRED_SYSTEMS CPP CUDA) # Options are: CPP, CUDA, TBB or OMP.
    add_subdirectory(/3rdparty/thrust thrust)
    thrust_create_target(ThrustCUDA HOST CPP DEVICE CUDA)

    thrust_update_system_found_flags() # required due to CMake scoping rules
    set(THRSUT_LIB_FROM_SOURCE ThrustCUDA)
    message(STATUS "THRUST VERSION: ${THRUST_VERSION}")
    message(STATUS "CUB VERSION: ${CUB_VERSION}")
    message(STATUS "THRUST_CUDA_FOUND: ${THRUST_CUDA_FOUND}")
    message(STATUS "CUDA_AVAILABLE: ${CUDA_AVAILABLE}")
else()
    set(THRSUT_LIB_FROM_SOURCE "")
    set(CUDA_AVAILABLE False)
    message(STATUS "No CUDA support")
endif()

# If you need to compile for debug mode, 
# change this to "Debug"
# set(CMAKE_BUILD_TYPE Debug)

# set_property(TARGET tgt PROPERTY CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD 17)

# enables configuration generation for `ctest` command
enable_testing()

################ 3rd Party Libraries ################
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
include_directories(${OpenCV_INCLUDE_DIRS})


################## Learning 3rd party libraries ##################
# eigen example
add_executable(eigenMatrix practices/eigenMatrix.cpp)
target_link_libraries (eigenMatrix Eigen3::Eigen)

# practice OpenCV Viz3d
# https://docs.opencv.org/4.5.3/d7/df9/tutorial_table_of_content_viz.html
add_executable(viz3d_practice practices/viz3d_practice.cpp)
target_link_libraries (viz3d_practice utils Eigen3::Eigen ${OpenCV_LIBS})

# opencv example
add_executable(imageBasics practices/imageBasics.cpp)
target_link_libraries(imageBasics ${OpenCV_LIBS})

# cuda library example
if(CUDA_AVAILABLE)
    add_subdirectory("practices/cuda_practice")
endif()

############ Shared Library for Shared Functionalities between Exercises ############
# utils library
add_subdirectory(utils)

########################### Exercises ###########################
###### exercise 01
add_subdirectory("camera_model")
add_executable(exercise01 exercise01.cpp)
target_link_libraries (exercise01 camera_model)

###### exercise 02
add_subdirectory("pnp_dlt")
add_executable(exercise02 exercise02.cpp)
target_link_libraries (exercise02 pnp_dlt)

###### exercise 03
add_subdirectory("keypoint_tracking")
add_executable(exercise03 exercise03.cpp)
target_link_libraries (exercise03 keypoint_tracking)

###### exercise 04
add_library(sift sift.cpp sift.hpp image_gradient.cpp image_gradient.hpp)
get_filename_component(SIFT_HEADER_DIR "./" ABSOLUTE)
target_include_directories(sift PUBLIC ${SIFT_HEADER_DIR})
target_link_libraries (sift utils)

add_executable(exercise04 exercise04.cpp)
target_link_libraries (exercise04 sift)

###### exercise 05
add_library(stereo_reconst stereo_reconst.cpp stereo_reconst.hpp)
get_filename_component(stereo_reconst_HEADER_DIR "./" ABSOLUTE)
target_include_directories(stereo_reconst PUBLIC ${stereo_reconst_HEADER_DIR})
target_link_libraries (stereo_reconst utils tbb)


add_executable(exercise05 exercise05.cpp)
target_link_libraries (exercise05 stereo_reconst)

###### exercise 06
add_library(two_view_geometry two_view_geometry.cpp two_view_geometry.hpp)
get_filename_component(two_view_geometry_HEADER_DIR "./" ABSOLUTE)
target_include_directories(two_view_geometry PUBLIC ${two_view_geometry_HEADER_DIR})
target_link_libraries (two_view_geometry utils tbb)

add_executable(exercise06 exercise06.cpp)
target_link_libraries (exercise06 two_view_geometry)


############################ Unit Tests ###########################
add_subdirectory(tests)