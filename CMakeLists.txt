cmake_minimum_required(VERSION 3.10)
project(CurveDeformer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Manually specify Eigen include directory (Homebrew installation path)
set(EIGEN3_INCLUDE_DIR "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3")
if(NOT EXISTS ${EIGEN3_INCLUDE_DIR})
    message(FATAL_ERROR "Eigen not found! Please install Eigen or specify the correct path.")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

# Add executable
add_executable(CurveDeformer main.cpp)

# Link OpenCV libraries
target_link_libraries(CurveDeformer ${OpenCV_LIBRARIES})

# Optional: Print found libraries for debugging
message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
message(STATUS "Eigen include dir: ${EIGEN3_INCLUDE_DIR}")
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
