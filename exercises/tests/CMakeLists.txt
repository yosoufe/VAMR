
####################### GTEST
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
# message ('${GTEST_BOTH_LIBRARIES}')

##################### Tests

### Libs
add_executable(test_matlab_like test_matlab_like.cpp)
target_link_libraries(test_matlab_like
    utils
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

add_test(
    NAME test_matlab_like
    COMMAND test_matlab_like
)

### exercise 04
add_executable(exercise04_tests test_sift.cpp)
target_link_libraries(exercise04_tests
    sift
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

add_test(
    NAME exercise04_tests
    COMMAND exercise04_tests
)


### exercise 05
add_executable(exercise05_tests test_stereo_reconst.cpp)
target_link_libraries(exercise05_tests
    stereo_reconst
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

add_test(
    NAME exercise05_tests
    COMMAND exercise05_tests
)

### exercise 06
add_executable(test_two_view_geometry test_two_view_geometry.cpp)
target_link_libraries(test_two_view_geometry
    two_view_geometry
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

add_test(
    NAME test_two_view_geometry
    COMMAND test_two_view_geometry
)

### exercise 07
add_executable(test_ransac test_ransac.cpp)
target_link_libraries(test_ransac
    ransac
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

add_test(
    NAME test_ransac
    COMMAND test_ransac
)