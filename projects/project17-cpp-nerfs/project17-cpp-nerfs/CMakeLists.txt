# CMakeList.txt : CMake project for project17-cpp-nerfs, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (project17-cpp-nerfs "project17-cpp-nerfs.cpp"  "nerf_math.h" "nerf_math.cpp" "nerf_defines.h" "nerf_data.h" "nerf_data.cpp" "vendor/stb_image_impl.cpp")

# Vendored single-header libraries: stb_image.h and nlohmann/json (json.hpp).
target_include_directories (project17-cpp-nerfs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET project17-cpp-nerfs PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
