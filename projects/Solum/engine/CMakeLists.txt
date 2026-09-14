# CMakeList.txt : CMake project for engine, include source and define
# project specific logic here.
#

# Found via the VULKAN_SDK environment variable the SDK installer sets.
find_package (Vulkan REQUIRED)

# Shaders are compiled to SPIR-V at build time and emitted as C arrays, so the
# engine has no shader files to find at runtime and no load path to get wrong.
if (NOT Vulkan_GLSLC_EXECUTABLE)
  message (FATAL_ERROR "glslc not found. It ships with the Vulkan SDK; set VULKAN_SDK.")
endif ()

set (SOLUM_SHADER_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file (MAKE_DIRECTORY "${SOLUM_SHADER_OUT_DIR}")

set (SOLUM_SHADER_HEADERS "")
foreach (shader "static_mesh.vert" "static_mesh.frag")
  set (shader_src "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}")
  set (shader_out "${SOLUM_SHADER_OUT_DIR}/${shader}.inl")
  add_custom_command (
    OUTPUT "${shader_out}"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -O -mfmt=c "${shader_src}" -o "${shader_out}"
    DEPENDS "${shader_src}"
    COMMENT "Compiling ${shader} to SPIR-V")
  list (APPEND SOLUM_SHADER_HEADERS "${shader_out}")
endforeach ()

set_source_files_properties (${SOLUM_SHADER_HEADERS} PROPERTIES HEADER_FILE_ONLY TRUE)

# The renderer knows nothing about the window system, so the GLFW runtime and
# the Qt editor can both drive it.
add_library (solum STATIC "sol_list.h" "sol_pool.h" "sol_pool.cpp" "sol_string.h" "sol_string.cpp" "sol_asset.h" "sol_asset.cpp" "sol_math.cpp" "sol_world.h" "sol_world.cpp" "sol_halfmesh.h" "sol_halfmesh.cpp" "sol_camera.h" "sol_camera.cpp" "sol_render.cpp" ${SOLUM_SHADER_HEADERS})
target_include_directories (solum PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories (solum PRIVATE "${SOLUM_SHADER_OUT_DIR}")

# sol_asset.cpp uses fopen, which MSVC deprecates in favour of its own fopen_s.
# The portable call is the right one to keep, so silence the deprecation rather
# than write non-standard code.
if (MSVC)
  target_compile_definitions (solum PRIVATE _CRT_SECURE_NO_WARNINGS)
endif ()
target_link_libraries (solum PUBLIC Vulkan::Vulkan vma)
set_property (TARGET solum PROPERTY CXX_STANDARD 20)

add_executable (engine "sol_engine.cpp")
target_link_libraries (engine PRIVATE solum glfw)
set_property (TARGET engine PROPERTY CXX_STANDARD 20)

# TODO: Add tests and install targets if needed.
