cmake_minimum_required(VERSION 3.20)

# vcpkg supplies raylib for the native build. Set before project() so the
# toolchain file is picked up on the first configure; -DCMAKE_TOOLCHAIN_FILE
# on the command line still wins, and the emscripten build ignores it.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND NOT EMSCRIPTEN)
    foreach(_vcpkg_root "$ENV{VCPKG_ROOT}" "C:/Projects/Programs/vcpkg" "C:/vcpkg")
        if(_vcpkg_root AND EXISTS "${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake")
            set(CMAKE_TOOLCHAIN_FILE "${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake"
                CACHE FILEPATH "vcpkg toolchain")
            break()
        endif()
    endforeach()
endif()

project(BasicGaussians LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(EMSCRIPTEN)
    # vcpkg has no emscripten raylib, so build it from source configured for the
    # web backend (GLFW is replaced by emscripten's own canvas/HTML5 input).
    include(FetchContent)
    set(PLATFORM Web CACHE STRING "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(CUSTOMIZE_BUILD ON CACHE BOOL "" FORCE)
    set(SUPPORT_FILEFORMAT_JPG ON CACHE BOOL "" FORCE)
    FetchContent_Declare(raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG 5.5
    )
    FetchContent_MakeAvailable(raylib)
else()
    find_package(raylib CONFIG REQUIRED)
endif()

# glm comes from source rather than vcpkg so the native and emscripten builds
# use the same headers -- it is header-only, so there is nothing to compile.
include(FetchContent)
set(GLM_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

file(GLOB_RECURSE BASIC_GAUSSIANS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

add_executable(BasicGaussians ${BASIC_GAUSSIANS_SOURCES})
target_include_directories(BasicGaussians PRIVATE src)
target_link_libraries(BasicGaussians PRIVATE raylib glm::glm)

target_compile_definitions(BasicGaussians PRIVATE
    GLM_ENABLE_EXPERIMENTAL          # gtx/ headers: quaternions, matrix decompose
    GLM_FORCE_EXPLICIT_CTOR)         # no silent vec4 -> vec3 conversions

if(MSVC)
    target_compile_options(BasicGaussians PRIVATE /MP /W4 /permissive- /utf-8)
    # Run from the source tree so relative asset paths resolve under F5.
    set_target_properties(BasicGaussians PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        PROPERTY VS_STARTUP_PROJECT BasicGaussians)
    source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src"
        FILES ${BASIC_GAUSSIANS_SOURCES})
endif()

if(EMSCRIPTEN)
    target_compile_definitions(BasicGaussians PRIVATE PLATFORM_WEB)
    # .html so emcc emits a page next to the .js/.wasm
    set_target_properties(BasicGaussians PROPERTIES SUFFIX ".html")
    target_link_options(BasicGaussians PRIVATE
        -sUSE_GLFW=3
        -sASYNCIFY
        -sALLOW_MEMORY_GROWTH=1
        -sSTACK_SIZE=1048576
        -sFORCE_FILESYSTEM=1
        # the page drives the demo through these instead of raylib's own overlay
        -sEXPORTED_RUNTIME_METHODS=ccall,cwrap
        --shell-file "${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html"
    )
endif()
