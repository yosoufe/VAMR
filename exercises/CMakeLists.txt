cmake_minimum_required(VERSION 3.0)
project(exercise01)

find_package (Eigen3 REQUIRED NO_MODULE)

# eigen example
add_executable(eigenMatrix eigenMatrix.cpp)
target_link_libraries (eigenMatrix Eigen3::Eigen)

# utils library
add_library(utils SHARED utils.cpp utils.hpp)
target_link_libraries(utils Eigen3::Eigen ${OpenCV_LIBS})


# opencv example
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(imageBasics imageBasics.cpp)
target_link_libraries(imageBasics ${OpenCV_LIBS})


# exercise 01
add_executable(exercise01 exercise01.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries (exercise01 utils Eigen3::Eigen ${OpenCV_LIBS})


# exercise 02
add_executable(exercise02 exercise02.cpp)
target_link_libraries (exercise02 utils Eigen3::Eigen ${OpenCV_LIBS})

# practice OpenCV Viz3d
# https://docs.opencv.org/4.5.3/d7/df9/tutorial_table_of_content_viz.html
add_executable(viz3d_practice viz3d_practice.cpp)
target_link_libraries (viz3d_practice utils Eigen3::Eigen ${OpenCV_LIBS})