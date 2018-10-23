cmake_minimum_required(VERSION 3.0)

if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )
set( 
    SOURCE
    ${SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VisionClass.cpp
    ${PARENT_DIRECTORY}
)
set(
  HEADERS
  ${HEADERS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include/VisionClass.hpp
  ${PARENT_DIRECTORY}
)

project( ComputerVision )
find_package( OpenCV REQUIRED )

add_executable( ComputerVision ${SOURCE} ${HEADERS} )
# target_include_directories( ComputerVision include/VisionClass.hpp )
target_link_libraries( ComputerVision ${OpenCV_LIBS} )

