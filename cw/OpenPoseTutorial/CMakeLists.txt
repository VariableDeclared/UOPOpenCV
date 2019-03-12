
cmake_minimum_required(VERSION 3.13)

project(OpenPoseTutorial)

# Require version without issue with empty Matrices...
find_package(OpenCV 3.4.3 REQUIRED)

add_executable(tutorial "main.cpp")

target_link_libraries(tutorial ${OpenCV_LIBS})