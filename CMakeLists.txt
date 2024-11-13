cmake_minimum_required(VERSION 3.28)
project(SIPADPBT)

set(CMAKE_CXX_FLAGS "/std:c++20 /EHsc /bigobj")
set(CMAKE_PREFIX_PATH "C:/module")
set(CMAKE_CXX_STANDARD 20)
add_executable(${PROJECT_NAME} main.cpp pch.cpp TimerCapture.h)
target_precompile_headers(${PROJECT_NAME} PUBLIC "pch.h")

include_directories(
    "C:/module/include"
)

# packages
set(PACKAGES "Matplot++" "OpenBLAS")
foreach(PACKAGE IN LISTS PACKAGES)
    find_package(${PACKAGE} REQUIRED)
    if(${PACKAGE}_FOUND)
        message("===== FOUND ${PACKAGE} Version ${${PACKAGE}_VERSION}")
    endif()
endforeach()

target_link_libraries(
    ${PROJECT_NAME}
    Matplot++::nodesoup 
    Matplot++::cimg 
    Matplot++::matplot
    OpenBLAS::OpenBLAS
)