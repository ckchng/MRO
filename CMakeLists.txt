

cmake_minimum_required (VERSION 3.0)
project (irotavg)

set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


set (CMAKE_CXX_STANDARD 11)

# Make CMake aware of the cmake folder for local FindXXX scripts,
# append rather than set in case the user has passed their own
# additional paths via -D.
list(APPEND CMAKE_MODULE_PATH "${irotavg_SOURCE_DIR}/ral/cmake")
include(UpdateCacheVariable)

message( "irotavg_SOURCE_DIR:    ${irotavg_SOURCE_DIR}" )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(src)
