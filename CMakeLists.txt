cmake_minimum_required(VERSION 3.17)
project(mlc_example)

set(CMAKE_CXX_STANDARD 14)

set(COMPILE_FLAGS "${COMPILE_FLAGS} -fno-objc-arc")

add_subdirectory(MLComputeCpp)

add_executable(${PROJECT_NAME}
    main.cpp)

target_link_libraries(${PROJECT_NAME}
    "-framework Foundation"
    "-framework MLCompute"
    mlc_lib)

