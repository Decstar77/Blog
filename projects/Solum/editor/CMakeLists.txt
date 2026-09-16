# CMakeList.txt : Qt editor shell around the engine renderer.
#

find_package (Qt6 QUIET COMPONENTS Widgets Gui Core)

if (NOT Qt6_FOUND)
  message (STATUS "Qt6 not found, skipping the editor target. "
                  "Point CMAKE_PREFIX_PATH at a Qt installation to build it.")
  return ()
endif ()

set (CMAKE_AUTOMOC ON)

add_executable (editor "sol_editor_main.cpp" "sol_editor_view.cpp" "sol_editor_import.cpp" "sol_editor_gizmo.h" "sol_editor_grid.h" "sol_editor_gizmo.cpp" "sol_editor_assets.h" "sol_editor_assets.cpp")
target_link_libraries (editor PRIVATE solum stb Qt6::Widgets Qt6::Gui Qt6::Core)
# The asset browser and importer work against the source tree's assets folder.
target_compile_definitions (editor PRIVATE SOLUM_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../assets")
# Qt's slots/signals/emit macros clash with engine names like Pool::slots, and
# would otherwise break any file that includes a Qt header before the engine's.
target_compile_definitions (editor PRIVATE QT_NO_KEYWORDS)
set_property (TARGET editor PROPERTY CXX_STANDARD 20)

# Qt's DLLs live outside the build tree, so stage them next to the exe.
if (WIN32)
  find_program (SOLUM_WINDEPLOYQT windeployqt HINTS "${Qt6_DIR}/../../../bin")
  if (SOLUM_WINDEPLOYQT)
    add_custom_command (TARGET editor POST_BUILD
      COMMAND "${SOLUM_WINDEPLOYQT}" --no-translations --no-compiler-runtime
              "$<TARGET_FILE:editor>"
      COMMENT "Staging Qt runtime next to editor.exe")
  endif ()
endif ()
