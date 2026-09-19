# CMakeList.txt : CMake project for project17-gaussian-renderer, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.18) # 3.18+ for CUDA_ARCHITECTURES and FindCUDAToolkit

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# C is still needed: the vendored glfw/glad targets are C.
project ("project17-gaussian-renderer" LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(GAUSSIAN_CUDA_ARCH "86-real" CACHE STRING "CUDA architectures to build device code for")

# --- Dependencies (vendored source, see vendor/) ---------------------------
find_package(OpenGL REQUIRED)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/glfw)

add_subdirectory(vendor/glm)

# glad generates the GL 4.6 core loader at configure time (needs python).
add_subdirectory(vendor/glad/cmake glad_cmake)
glad_add_library(glad_gl_core_46 REPRODUCIBLE API gl:core=4.6)

# Add source to this project's executable.
add_executable (project17-gaussian-renderer
  "gs_defines.h"
  "gs_list.h"
  "gs_splats.h"
  "gs_splats.cpp"
  "gs_scene.h"
  "gs_scene.cpp"
  "main.cpp"
  "gs_window.cpp"
  "gs_glrenderer.cpp"
  "cuda_render.cu"
)

# cub (CCCL, shipped with the toolkit) hard-errors against MSVC's traditional preprocessor.
target_compile_options(project17-gaussian-renderer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)

target_link_libraries(project17-gaussian-renderer PRIVATE
  glfw
  glad_gl_core_46
  glm::glm
  OpenGL::GL
  CUDA::cudart
)

# Let .cu translation units include the vendored headers and resolve "cuda_render.h".
target_include_directories(project17-gaussian-renderer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(project17-gaussian-renderer PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  # sm_86 is the RTX 30-series (Ampere) this is developed on; "-real" emits the
  # cubin only. Without an exact-arch cubin the driver has to JIT our PTX, and a
  # driver older than the toolkit fails that with "unsupported toolchain".
  # Override with -DGAUSSIAN_CUDA_ARCH=<arch> to run on other hardware.
  CUDA_ARCHITECTURES "${GAUSSIAN_CUDA_ARCH}"
  # nvcc cannot take /ZI, so opt this target out of the Edit-and-Continue
  # debug format CMP0141 selects above.
  MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>"
)

# TODO: Add tests and install targets if needed.
