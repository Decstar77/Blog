# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Solum is a small C++20 Vulkan engine plus a Qt6 level editor. It lives inside the blog repo
(`C:/Projects/2025/Blog`, git root two levels up) as one of the standalone, non-runner projects;
the blog-level CLAUDE.md does not apply to anything in here beyond that.

## Building

CMake + Ninja + MSVC (`cl.exe`), Windows only. Presets are in `CMakePresets.json`
(`x64-debug`, `x64-release`, `x86-*`), building into `out/build/<preset>`. Run from a VS
developer environment (the presets reference `$env{VSINSTALLDIR}`):

```
cmake --preset x64-debug
cmake --build out/build/x64-debug                 # everything
cmake --build out/build/x64-debug --target editor # or: engine, solum
```

`out/build/` also contains ad-hoc build dirs (`agent-*`, `master`). Use a separate binary dir if you
need to build without touching the user's `x64-debug` tree.

External requirements:
- **Vulkan SDK** (`VULKAN_SDK` set). `find_package(Vulkan)` is required, and so is `glslc`: shaders
  in `engine/shaders/` compile at build time to C arrays (`<build>/engine/shaders/*.inl`) that
  `sol_render.cpp` includes. No shader files are loaded at runtime. To add a shader, add it to the
  `foreach` list in `engine/CMakeLists.txt`.
- **Qt6** (Widgets/Gui/Core). Defaults to `C:/Qt/6.11.1/msvc2022_64` if `CMAKE_PREFIX_PATH` is unset.
  If Qt isn't found the `editor` target is **silently skipped** (a STATUS message only). A post-build
  `windeployqt` step copies Qt DLLs next to `editor.exe`.
- Vendored in `vendor/`: glfw (built via `add_subdirectory`), VMA and stb (each compiled in its own
  single-TU static lib, `vma_impl.cpp` / `stb_image_impl.cpp`), `json.hpp`.

There are no tests and no lint step. Formatting is `.clang-format` (4-space indent, no column limit,
spaces inside parentheses: `Foo( a, b )`, `Type * ptr`).

## Targets and layering

- **`solum`** (static lib, `engine/`): renderer, world, half-edge mesh, camera, assets, containers.
  Links only Vulkan + VMA. It must not know about windowing: GLFW and Qt both drive it.
- **`engine`** (`engine/sol_engine.cpp`): minimal GLFW runtime. It creates a window, loads a hardcoded
  absolute asset path (falls back to a checkerboard), and draws with a fly camera.
- **`editor`** (`editor/`): Qt app. Links `solum` + `stb`. **Only the editor may link stb or decode
  source image formats.** The engine reads only its own `.stex`/`.meta` asset files.
  `SOLUM_ASSET_DIR` is a compile definition pointing at `assets/` in the source tree.
  - `editor --import <sourcePath> <outputDirectory> <assetName>` is a headless import mode (no Qt app,
    no Vulkan). It writes the asset pair and verifies it round-trips via `TextureAssetLoad`.
  - Built with `QT_NO_KEYWORDS`, so use `Q_SIGNALS`/`Q_SLOTS`/`Q_EMIT`. Bare `slots`/`emit` clash with
    engine names like `PoolSlots`.

New `.cpp` files must be added to the explicit source lists in `engine/CMakeLists.txt` or
`editor/CMakeLists.txt`. There is no globbing.

## Code style / conventions

The engine is written in a C-like style, all inside `namespace sol`:
- Plain structs with no constructors (`StringView` is the one deliberate exception), zero-initialised
  with `= {}`, operated on by free functions prefixed with the type: `RendererDrawFrame( &r )`,
  `WorldAddPrimitive( world, r, ... )`, `ListAdd( list, v )`. Where zero isn't a valid default there's
  an explicit `XDefault()` (`TransformDefault`, `RenderMaterialDefault`, `TextureMetaDefault`).
- No STL containers and no exceptions. Errors come back as `bool` returns or null handles. Use the
  engine's own types from `sol_defines.h` (`i32`, `f32`, `u8`...), `List<T>` (`sol_list.h`, a
  realloc-backed POD array that must be `ListFree`d manually), and strings from `sol_string.h`
  (`StringView` as the parameter type everywhere, `FixedString<N>`/`SmallString`/`LargeString`,
  `HeapString`). Utility macros are still named `SPLATS_*` (`SPLATS_ARRAY_COUNT`, `SPLATS_UNUSED`).
- Comments explain *why* (constraints, Vulkan rules, lifetime hazards) and sit on struct fields and
  API declarations in headers. Match that density when adding code.

## Architecture notes that span files

- **Handles vs indices.** `Pool<T>` / `Handle<T>` (`sol_pool.h`) are generational handles. A stale
  handle resolves to null, not to a different object. The renderer hands out `RenderMeshHandle` and
  `RenderTextureHandle`. Pointers from `RendererGetStaticMesh`/`PoolGet` are only good until the next
  add/remove, so don't store them. A null or stale texture handle falls back to the renderer's white
  1x1 texture.
- **World vs renderer ownership.** `World` (`sol_world.h`) owns the authored scene:
  `List<Primitive>`, each with a `HalfMesh`, material, decomposed `Transform` (position / Euler
  radians / scale, kept decomposed to avoid gizmo drift), and a `RenderMeshHandle`. The renderer owns
  the triangulated GPU meshes. Editing a half-mesh (e.g. `WorldSetVertexPosition`) doesn't touch the
  GPU. Call `WorldRebuildPrimitive` afterwards; the editor batches this once per frame via
  `editGeometryDirty`. Primitives are addressed by `i32` index into the list, so removal shifts indices.
  Use `WorldRemapPrimitive` to fix up any index you're holding.
- **Device-idling calls are stalls.** `RendererCreateStaticMesh`/`Destroy*`, `RendererSetGridSpacing`,
  `RendererSetEditOverlay`, `RendererSetGizmoGeometry` all block or idle the device. They belong on
  load, selection change, or key press, never per frame. Per-frame state is CPU-only: mesh
  `transform`/`tint`, `RendererSetViews`, `RendererSetGizmoDraw`.
- **Rendering model.** One swapchain, one render pass, one static-mesh pipeline (plus grid/line/point
  variants sharing its layout and `StaticMeshVertex` format). Per-draw data is `StaticMeshPush` (MVP,
  tint, point size), which must stay within 128 bytes of push constants. Selection highlighting is
  just a tint. The scene is drawn once per `RenderView` (up to 4 normalized viewport rects, each with
  its own view-projection). The editor uses two: a perspective `FlyCamera` pane and a top-down
  `OrthoCamera` pane. Draw order per view: grid, meshes, then edit overlay and gizmo with depth test off.
- **Renderer startup is split** into `RendererCreateInstance` then `RendererStartup(surface, ownsSurface)`,
  because Qt needs the `VkInstance` before it can create a surface. In the editor Qt owns the surface,
  so `RendererShutdownDevice` must run while the window is still alive. `sol_editor_main.cpp` passes the
  engine's instance to `QVulkanInstance::setVkInstance`.
- **Editor interaction** lives in `VulkanView` (`editor/sol_editor_view.{h,cpp}`, a `QWindow` wrapped
  by `createWindowContainer`). It handles picking, drag-to-create planes in the top pane, Tab edit mode
  (locks selection, shows the half-mesh cage, vertex picking in screen space), and T/R gizmos
  (`sol_editor_gizmo.*`, which computes each drag from the drag-start transform). The status-bar text
  in `sol_editor_main.cpp` lists the controls; keep it in sync when adding bindings.
- **Assets** (`sol_asset.h`) are pairs: `<name>.meta` is a text `key = value` sidecar (source path,
  binary filename, format/filter/wrap), and `<name>.stex` is a 32-byte `TextureBinHeader` (magic
  `'SOLT'`) followed by raw RGBA8 pixels. The header layout is append-only: bump `kTextureVersion`
  rather than reordering. The editor's asset browser lists `.meta` files straight from disk.
