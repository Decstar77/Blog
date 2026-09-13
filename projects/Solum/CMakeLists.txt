# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.16)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("Solum")

# vendor/glfw declares cmake_minimum_required(VERSION 3.4...), which CMake 4.x
# refuses outright. This tells it to treat anything older than 3.5 as 3.5.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
add_subdirectory ("vendor/glfw")

# stb lives in its own target so only the editor links it. The engine must never
# depend on this: decoding source art is the importer's job, not the runtime's.
add_library (stb STATIC "vendor/stb_image_impl.cpp")
target_include_directories (stb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
set_property (TARGET stb PROPERTY CXX_STANDARD 20)

# stb_image_write uses sprintf, which MSVC deprecates. This is vendored code we
# do not want to patch, so silence it for this target only.
if (MSVC)
  target_compile_definitions (stb PRIVATE _CRT_SECURE_NO_WARNINGS)
endif ()

# VulkanMemoryAllocator is a single header; this target is the one TU that
# compiles its implementation, so every consumer just includes the header.
find_package (Vulkan REQUIRED)
add_library (vma STATIC "vendor/vma_impl.cpp")
target_include_directories (vma PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries (vma PUBLIC Vulkan::Vulkan)
set_property (TARGET vma PROPERTY CXX_STANDARD 20)

# Vendored code we do not want to patch: VMA trips MSVC's unreferenced-parameter
# and nameless-struct warnings, and uses functions MSVC deprecates.
if (MSVC)
  target_compile_definitions (vma PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_options (vma PRIVATE /w)
endif ()

# Where find_package looks for Qt6 when the caller has not said otherwise.
if (NOT DEFINED CMAKE_PREFIX_PATH AND EXISTS "C:/Qt/6.11.1/msvc2022_64")
  set(CMAKE_PREFIX_PATH "C:/Qt/6.11.1/msvc2022_64")
endif()

# Include sub-projects.
add_subdirectory ("engine")
add_subdirectory ("editor")
