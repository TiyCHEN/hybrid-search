cmake_minimum_required(VERSION 3.15)
project(HYBRID)

set(CMAKE_CXX_STANDARD 17)

# include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-mavx" COMPILER_SUPPORTS_MAVX)
# if(COMPILER_SUPPORTS_MAVX)
#     message("useavx")
#     add_compile_options(-mavx)
#     add_definitions(-DUSE_AVX)
# endif()


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -pthread -march=native")
set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g2 -ggdb -pthread -Wall -DINFO")

# Includes
set(HYBRID_SRC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(HYBRID_THIRD_PARTY_INCLUDE_DIR
        ${PROJECT_SOURCE_DIR}/third_party
)

include_directories(${HYBRID_SRC_INCLUDE_DIR} ${HYBRID_THIRD_PARTY_INCLUDE_DIR})

# find_package(Boost REQUIRED)
# if (Boost_FOUND)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Boost_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Boost_CXX_FLAGS}")
#     include_directories(${Boost_INCLUDE_DIRS})
# else()
#     message(FATAL_ERROR "Boost dynamic-bitset is required")
# endif()

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP is not found")
endif()

# Compile flags
# add_definitions(-lboost -ltcmalloc_minimal)

add_executable(main main.cc)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx" COMPILER_SUPPORTS_MAVX)
if(COMPILER_SUPPORTS_MAVX)
    message("Compiler supports AVX. USE_AVX will be defined.")
    add_compile_options(-mavx2)
    target_compile_definitions(main PUBLIC USE_AVX)
endif()

# target_link_libraries(main src)
