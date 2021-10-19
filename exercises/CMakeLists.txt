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
)
target_link_libraries(utils Eigen3::Eigen ${OpenCV_LIBS})


########################### Exercises ###########################
###### exercise 01
add_executable(exercise01 exercise01.cpp)
target_link_libraries (exercise01 utils)

###### exercise 02
add_executable(exercise02 exercise02.cpp)
target_link_libraries (exercise02 utils)

###### exercise 03
add_executable(exercise03 exercise03.cpp)
target_link_libraries (exercise03 utils Eigen3::Eigen ${OpenCV_LIBS})

###### exercise 03
add_library(sift sift.cpp sift.hpp)
get_filename_component(SIFT_HEADER_DIR "./" ABSOLUTE)
target_include_directories(sift PUBLIC ${SIFT_HEADER_DIR})
target_link_libraries (sift utils)

add_executable(exercise04 exercise04.cpp)
target_link_libraries (exercise04 sift)



############################ Unit Tests ###########################
add_subdirectory(tests)