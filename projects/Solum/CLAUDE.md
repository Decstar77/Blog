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

There are no tests in the repo and no lint step. Formatting is `.clang-format` (4-space indent, no column limit,
spaces inside parentheses: `Foo( a, b )`, `Type * ptr`).

## Targets and layering

- **`solum`** (static lib, `engine/`): renderer, brushes and maps (`sol_brush`, `sol_map`), world,
  half-edge mesh, camera, assets, containers. Links only Vulkan + VMA. It must not know about
  windowing: GLFW and Qt both drive it. `World`/`HalfMesh` and the renderer's edit-overlay and border
  APIs are no longer used by the editor, which builds with brushes.
- **`engine`** (`engine/sol_engine.cpp`): minimal GLFW runtime. It creates a window, loads a hardcoded
  absolute asset path (falls back to a checkerboard), and draws with a fly camera.
- **`editor`** (`editor/`): Qt app, a TrenchBroom-style brush editor. Links `solum` + `stb`. **Only the
  editor may link stb or decode source image formats.** The engine reads only its own
  `.stex`/`.meta`/`.smap` files. `SOLUM_ASSET_DIR` is a compile definition pointing at `assets/` in
  the source tree; maps default to `assets/maps/`.
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
- **Brushes are planes.** `Brush` (`sol_brush.h`) is a convex solid: a `List<BrushFace>` of outward
  planes, each with a `FaceTexture` (material name, offset/scale/rotation, projected paraxially from
  world space). Corners and polygons (`Brush::points`, per-face `firstPoint`/`pointCount`) are derived
  by `BrushRebuild`, which also drops redundant planes, so face indices can shift after any rebuild.
  Every edit goes through the planes: move a face = change its distance, vertex edits rebuild via
  `BrushCreateHull`, CSG is `BrushClipBehind`/`BrushSubtract`. Geometry is solved in doubles and
  corners near a 1/1024 lattice are snapped onto it, which is what keeps grid-aligned brushes exact.
  `BrushFlags`/`BrushFaceFlags` (selected, hidden) are editor state carried on the brush so undo
  snapshots restore selection; the `.smap` format (`sol_map.h`, text, planes only) does not save them.
- **Editor document.** `EditorDoc` (`editor/sol_editor_doc.h`) owns the `Map` and a whole-map snapshot
  undo stack. Edits are bracketed: `DocBeginEdit` → change → `DocEdited`, or `DocAbandonEdit` to put
  the snapshot back. A drag begins once, rewrites the map from its drag-start copies on every move
  (`DocTouch`), and settles on release, so a drag is one undo step. Brushes are addressed by index;
  nothing adds or removes brushes mid-drag, which is what keeps the drag's indices valid.
- **Device-idling calls are stalls.** `RendererCreateStaticMesh`/`Destroy*`, `RendererSetGridSpacing`,
  `RendererSetEditOverlay`, `RendererSetGizmoGeometry` all block or idle the device. They belong on
  load, selection change, or key press, never per frame. Per-frame state is CPU-only: mesh
  `transform`/`tint`, `RendererSetViews`, `RendererSetGizmoDraw`, and `RendererSetStream`.
- **Render streams** are how anything that changes under the mouse is drawn. `RendererSetStream`
  copies vertices + `RenderBatch`es into one of three streams (background, world, overlay); the
  renderer re-uploads a stream into a host-visible buffer per frame in flight only when its version
  moved, after that slot's fence. No stalls, ever. Each batch picks a `RenderBatchKind` pipeline
  (depth-biased solid, translucent, depth-tested lines, on-top lines/points), a texture, a tint, a
  `viewMask` (bit per view) and optionally `screenSpace` (clip-space vertices, for pane frames and
  rubber bands). The editor draws all brushes through the world stream (`sol_editor_draw.cpp`,
  rebuilt only when `doc.version` or a preview changes) and tool feedback through the overlay stream
  (rebuilt every frame).
- **Rendering model.** One swapchain, one render pass, one static-mesh shader pair shared by every
  pipeline, all using the `StaticMeshVertex` format. Per-draw data is `StaticMeshPush` (MVP, tint,
  point size), which must stay within 128 bytes of push constants. The fragment shader lights with a
  fixed key light; overlay geometry uses `kOverlayNormal` (the light direction) so its colours come
  through unshaded. The scene is drawn once per `RenderView` (up to 4 normalized viewport rects).
  Draw order per view: grid (unless `hideGrid`), background stream, static meshes, world stream, edit
  overlay, overlay stream, gizmo, then the optional surface border.
- **Renderer startup is split** into `RendererCreateInstance` then `RendererStartup(surface, ownsSurface)`,
  because Qt needs the `VkInstance` before it can create a surface. In the editor Qt owns the surface,
  and `QWindowContainer` destroys it *before* deleting the view, so `VulkanView` shuts the device down
  on `QPlatformSurfaceEvent::SurfaceAboutToBeDestroyed`, not in its destructor. `sol_editor_main.cpp`
  passes the engine's instance to `QVulkanInstance::setVkInstance`.
- **Editor layout.** `VulkanView` (a `QWindow` in a `createWindowContainer`) is split across
  `sol_editor_view.cpp` (panes, cameras, frame building, keys, commands, files, inspector hooks) and
  `sol_editor_tools.cpp` (every left-button interaction per `EditorTool`, keyboard edits, overlays).
  Pane slot 0 is always perspective and 1-3 are top/front/side ortho; layouts only resize slots (hidden
  panes get zero size), so the fixed `kMask3D`/`kMask2D` view masks stay valid. Menus are built from
  `EditorCommand` tables in `sol_editor_main.cpp` and call `VulkanView::Command`; keys are handled in
  `HandleKey` because Qt shortcuts don't reach the embedded window. Bindings live in three places that
  must agree: `HandleKey`, the menu tables, and the `kControlsHelp` sheet (Help > Controls). The rotate
  tool reuses `sol_editor_gizmo.*`; 3D-view creation and orbit/dolly still use `EditorGrid` (y = 0).
- **Assets** (`sol_asset.h`) are pairs: `<name>.meta` is a text `key = value` sidecar (source path,
  binary filename, format/filter/wrap), and `<name>.stex` is a 32-byte `TextureBinHeader` (magic
  `'SOLT'`) followed by raw RGBA8 pixels. The header layout is append-only: bump `kTextureVersion`
  rather than reordering. The editor's asset browser lists `.meta` files straight from disk, with
  thumbnails read from the `.stex`; clicking one applies it to the selection and makes it the material
  new brushes wear. Faces store the asset name relative to `assets/`, without the extension.
