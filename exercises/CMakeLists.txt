cmake_minimum_required(VERSION 3.0)
project(exercise01)

find_package (Eigen3 REQUIRED NO_MODULE)

# eigen example
add_executable(eigenMatrix eigenMatrix.cpp)
target_link_libraries (eigenMatrix Eigen3::Eigen)


# opencv example
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(imageBasics imageBasics.cpp)
target_link_libraries(imageBasics ${OpenCV_LIBS})


# exercise 01
add_executable(exercise01 exercise01.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries (exercise01 Eigen3::Eigen ${OpenCV_LIBS})
