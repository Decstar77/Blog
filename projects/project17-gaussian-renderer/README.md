# Gaussian Splat Renderer

A real-time 3D Gaussian Splatting rasterizer written from scratch in CUDA, with an OpenGL window
for presentation. It loads a trained `.ply` capture (the format the INRIA training code exports)
and flies a camera through it.

![The renderer running on the built-in demo scene](recording.gif)

No splatting library is used. The projection, tiling, sorting and compositing are all in
`cuda_render.cu`; OpenGL only owns the window and blits the CUDA-written texture to the screen.

## How it renders a frame

Splatting is order-dependent alpha compositing over millions of primitives, so the whole design is
about getting them into per-pixel depth order cheaply. The frame runs as four GPU passes.

**1. Preprocess** (`preprocess_splats`) projects every splat once. The centre goes through a pinhole
projection; the 3D covariance `R S SᵀRᵀ` is pushed through a local affine approximation of that
projection (the EWA method) to get a 2D covariance, which is inverted into a conic so the render
pass can evaluate a Gaussian with three multiplies. The larger eigenvalue of the 2D covariance is
the variance along the splat's long axis, so 3σ of it bounds the on-screen footprint in every
direction. That bound is clipped to the 16×16 tile grid, and the pass records how many tiles the
splat covers.

**2. Duplicate** (`duplicate_splats`) expands each splat into one entry per tile it touches. A
prefix sum over the per-splat tile counts gives each splat a private slice of the output array, so
the expansion needs no atomics. Each entry gets a 64-bit key: the tile id in the high 32 bits, the
float depth's raw bits in the low 32. Positive floats compare correctly as integers, so one radix
sort over that key groups entries by tile *and* orders them by depth within each tile at the same
time. Only the bits the tile id actually occupies are sorted above the depth bits.

**3. Tile ranges** (`identify_tile_ranges`) walks the sorted keys and records where each tile's run
of entries starts and ends.

**4. Composite** (`render_tiles`) runs one block per tile and one thread per pixel. The block
cooperatively loads its tile's splats into shared memory in 256-entry batches and each thread walks
them front to back, accumulating colour and attenuating transmittance. A thread that saturates
stops accumulating but keeps reaching the barriers, so the early exit is a block-wide
`__syncthreads_count` rather than a per-thread `break`. The result is written straight into an
OpenGL texture through CUDA/GL interop, so the pixels never travel back through host memory.

## Building

The dependencies (GLFW, glm, glad) are git submodules, so clone recursively:

```
git clone --recurse-submodules <repo>
cd projects/project17-gaussian-renderer
```

If you already cloned without `--recurse-submodules`:

```
git submodule update --init --recursive
```

Then configure and build. The presets use Ninja and `cl.exe`, so run these from a Visual Studio
developer command prompt (or any shell where `VsDevCmd.bat -arch=amd64` has been sourced):

```
cmake --preset x64-release
cmake --build out/build/x64-release
```

Requirements: the CUDA Toolkit (built and tested against 13.3), MSVC with C++20, and Python on
PATH (glad generates the GL 4.6 loader at configure time).

Device code is built for `sm_86` by default, which is what this was developed on. On other
hardware pass the arch you have:

```
cmake --preset x64-release -DGAUSSIAN_CUDA_ARCH=89-real
```

## Running

```
project17-gaussian-renderer.exe [path/to/scene.ply] [--flip-y]
```

With no arguments it builds a procedural demo scene

Given a `.ply`, the camera is framed automatically from the splat cloud's centre and spread. Pass
`--flip-y` for captures stored Z-up.

Controls: hold right mouse to look, `WASD` to move, `Space` / `Left Ctrl` for up and down, `Shift`
to move faster, `Esc` to quit.

## Known limitations

- **Degree-0 spherical harmonics only.** The `f_rest_*` coefficients are read past rather than
  decoded, so surfaces have no view-dependent shading. This is the visible quality gap against
  reference implementations.
- **No performance instrumentation.** There is no frame timer or per-pass GPU timing yet, so there
  are no numbers here to quote.
- **One synchronising readback per frame.** The total entry count comes back to the host after the
  prefix sum so the sort can be sized, which stalls the pipeline once a frame.
- **Sorted by splat centre, per tile.** This is what the reference implementation does, and it
  shares the same artefact: splats can pop as the camera moves and two centres swap order.
- **Fixed 16×16 tiles.** Chosen so the tile is also the shared-memory batch size. Other sizes are
  not benchmarked.
