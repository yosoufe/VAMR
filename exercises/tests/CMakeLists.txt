
####################### GTEST
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
# message ('${GTEST_BOTH_LIBRARIES}')

##################### Tests

### utils
add_executable(test_utils test_utils.cpp)
target_link_libraries(test_utils
    utils
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

### exercise 03
add_executable(test_exercise03 test_keypoint_tracking.cpp)
target_link_libraries(test_exercise03
    keypoint_tracking
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

add_test(
    NAME test_exercise03
    COMMAND test_exercise03
)


### exercise 04
add_executable(test_exercise04 test_sift.cpp)
target_link_libraries(test_exercise04
    sift
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

add_test(
    NAME test_exercise04
    COMMAND test_exercise04
)


### exercise 05
add_executable(test_exercise05 test_stereo_reconst.cpp)
target_link_libraries(test_exercise05
    stereo_reconst
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

add_test(
    NAME test_exercise05
    COMMAND test_exercise05
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