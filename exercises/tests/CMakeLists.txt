
####################### GTEST
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
# message ('${GTEST_BOTH_LIBRARIES}')

##################### Tests

# exercise 04
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


# exercise 05
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