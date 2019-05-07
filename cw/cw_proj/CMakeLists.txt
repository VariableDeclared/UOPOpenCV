cmake_minimum_required(VERSION 3.13)

project("CompVis Coursework")

find_package(OpenCV 3.4.5 REQUIRED)

add_executable(CompVisCoursework "main.cpp")


target_link_libraries(CompVisCoursework ${OpenCV_LIBS})
