cmake_minimum_required(VERSION 3.0)
project(VAMR)

# set_property(TARGET tgt PROPERTY CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD 17)

# enables configuration generation for `ctest` command
enable_testing()

################ 3rd Party Libraries ################
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# CvPlot, change it if you have cloned it in different location
include_directories("/3rdparty/cv-plot/CvPlot/inc")



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



############ Shared Library for Shared Functionalities between Exercises ############
# utils library
add_library(utils SHARED
    utils.cpp
    utils.hpp
    folder_manager.cpp
    folder_manager.hpp
    matlab_like.cpp # Matlab like functions
)
target_link_libraries(utils Eigen3::Eigen ${OpenCV_LIBS})
get_filename_component(utils_HEADER_DIR "./" ABSOLUTE)
target_include_directories(utils PUBLIC ${utils_HEADER_DIR})


########################### Exercises ###########################
###### exercise 01
add_executable(exercise01 exercise01.cpp)
target_link_libraries (exercise01 utils)

###### exercise 02
add_executable(exercise02 exercise02.cpp)
target_link_libraries (exercise02 utils)

###### exercise 03
add_executable(exercise03 exercise03.cpp)
target_link_libraries (exercise03 utils)

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

###### exercise 07

add_library(ransac ransac.cpp ransac.hpp)
get_filename_component(ransac_HEADER_DIR "./" ABSOLUTE)

target_include_directories(ransac PUBLIC ${ransac_HEADER_DIR})
target_link_libraries (ransac utils tbb)

add_executable(exercise07 exercise07.cpp)
target_link_libraries (exercise07 ransac)


############################ Unit Tests ###########################
add_subdirectory(tests)