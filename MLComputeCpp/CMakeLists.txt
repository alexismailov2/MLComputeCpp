cmake_minimum_required(VERSION 3.17)
project(mlc_lib)

set(CMAKE_CXX_STANDARD 14)
set(COMPILE_FLAGS "${COMPILE_FLAGS} -fno-objc-arc")
set(CMAKE_CXX_FLAGS "-x objective-c++")

file(GLOB CPP_MLCCOMPUTE_SOURCES *.h *.mm)

add_library(${PROJECT_NAME}
    ${CPP_MLCCOMPUTE_SOURCES})
