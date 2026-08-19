cmake_minimum_required(VERSION 3.20)
project(SphericalHarmonics LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_library(tinyexr STATIC
    vendor/tinyexr/tinyexr_impl.cpp
    vendor/tinyexr/deps/miniz/miniz.c
)
target_include_directories(tinyexr PUBLIC
    vendor/tinyexr
    vendor/tinyexr/deps/miniz
)

add_executable(SphericalHarmonics src/main.cpp)
target_link_libraries(SphericalHarmonics PRIVATE raylib tinyexr)

if(EMSCRIPTEN)
    target_compile_definitions(SphericalHarmonics PRIVATE PLATFORM_WEB)
    # .html so emcc emits a page next to the .js/.wasm/.data
    set_target_properties(SphericalHarmonics PROPERTIES SUFFIX ".html")
    target_link_options(SphericalHarmonics PRIVATE
        -sUSE_GLFW=3
        -sASYNCIFY                       # tinyexr/raylib do blocking work at startup
        -sALLOW_MEMORY_GROWTH=1
        -sSTACK_SIZE=1048576
        -sFORCE_FILESYSTEM=1
        # the page drives the demo through these instead of raylib's own overlay
        -sEXPORTED_RUNTIME_METHODS=ccall,cwrap
        --preload-file "${CMAKE_CURRENT_SOURCE_DIR}/malagas_ferry_road_1k.exr@malagas_ferry_road_1k.exr"
        --shell-file "${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html"
    )
endif()
