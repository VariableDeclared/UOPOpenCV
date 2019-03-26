cmake_minimum_required(VERSION 3.13)

project("Digits Tutorial")

find_package(OpenCV 3.4.3 REQUIRED)

add_executable(HOGTutorial "DigitsTutorial.cpp")


target_link_libraries(HOGTutorial ${OpenCV_LIBS})

