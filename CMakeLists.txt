cmake_minimum_required(VERSION 3.10)
project(PrivacyProtector)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find OpenCV
find_package(OpenCV 4.10 REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(privacy_protector main.cpp)

# Link OpenCV libraries
target_link_libraries(privacy_protector ${OpenCV_LIBS})
