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

# Where find_package looks for Qt6 when the caller has not said otherwise.
if (NOT DEFINED CMAKE_PREFIX_PATH AND EXISTS "C:/Qt/6.11.1/msvc2022_64")
  set(CMAKE_PREFIX_PATH "C:/Qt/6.11.1/msvc2022_64")
endif()

# Include sub-projects.
add_subdirectory ("engine")
add_subdirectory ("editor")
