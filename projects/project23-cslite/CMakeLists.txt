# CMakeList.txt : CMake project for ProximalPolicyOptim, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("ProximalPolicyOptim")

# raylib, fetched and built from source at configure time
include(FetchContent)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  raylib
  GIT_REPOSITORY https://github.com/raysan5/raylib.git
  GIT_TAG 5.5
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(raylib)

# Add source to this project's executable.
add_executable (ProximalPolicyOptim "Main.cpp" "CSPPO.h" "CSPPO.cpp" "ProximalPolicyOptim.cpp" "ProximalPolicyOptim.h" "Tensor.h" "Tensor.cpp" "Network.h" "Network.cpp" "CSGame.h" "CSGame.cpp" "MemoryArena.h" "MemoryArena.cpp" )
target_link_libraries(ProximalPolicyOptim PRIVATE raylib)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ProximalPolicyOptim PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
