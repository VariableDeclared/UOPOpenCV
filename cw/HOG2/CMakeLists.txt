cmake_minimum_required(VERSION 3.13)

project("HOG2")

find_package(OpenCV 3.4.3 REQUIRED)

add_executable(HOG2 "main.cpp")


target_link_libraries(HOG2 ${OpenCV_LIBS})
