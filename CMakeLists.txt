cmake_minimum_required(VERSION 3.21)
project(mag)

set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(Boost REQUIRED)
if (Boost_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Boost_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Boost_CXX_FLAGS}")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost dynamic-bitset is required")
endif()

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    include_directories(${OpenMP_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenMP is required")
endif()

# Compile flags

add_subdirectory(test)
add_definitions(-std=c++17 -O3 -lboost -ltcmalloc_minimal -lprofiler -march=native -Wall -DINFO -mavx)

list(
    APPEND CPP_SOURCES
    ./src/index_map.cpp
)
add_library(${PROJECT_NAME} ${CPP_SOURCES})
add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})


