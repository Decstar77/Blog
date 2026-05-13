package logic

// World-space MSDF text. Generated atlases live under `res/fonts/` (PNG + JSON
// produced by msdf-atlas-gen). The logic layer keeps font handles opaque; the
// platform layer owns the GL texture and shader.

Font_Handle :: distinct u32

INVALID_FONT :: Font_Handle(0)

// Default MSDF distance range (must match `-pxrange` used at generation).
MSDF_DEFAULT_PX_RANGE :: f32(4)
