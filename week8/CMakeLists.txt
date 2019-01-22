cmake_minimum_required(VERSION 3.0)
project( ComputerVision C CXX )

if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )
string(TIMESTAMP TIME "%Y%m%d")
set( PROJECT_EXECUTABLE_NAME "ComputerVision_${TIME}" )

set( 
    SOURCE
    ${SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FaceAnalysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cvtutorial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cv_util.cpp
    ${PARENT_DIRECTORY}
)
set(
  HEADERS
  ${HEADERS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include/FaceAnalysis.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cvtutorial.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cv_util.hpp
  ${PARENT_DIRECTORY}
)

set( CMAKE_BUILD_TYPE Debug)
# add_compile_definitions(COMPVIS_DEBUG
# set(CMAKE_C_FLAGS_DEBUG "-DCOMPVIS_DEBUG")
# file(GLOB HEADERS
#     "*.hpp"
# )

include_directories(include)
find_package( OpenCV 3.4.5 REQUIRED )


add_executable( ${PROJECT_EXECUTABLE_NAME} ${SOURCE} ${HEADERS} )
set_target_properties( ${PROJECT_EXECUTABLE_NAME} PROPERTIES LINKER_LANGUAGE CXX)
# target_include_directories( ComputerVision PRIVATE include/VisionClass.hpp )
target_link_libraries( ${PROJECT_EXECUTABLE_NAME} ${OpenCV_LIBS} )

