package logic

import "core:math"

Mesh_Handle :: distinct u32

INVALID_MESH :: Mesh_Handle(0)

Renderer :: struct {
    create_mesh  : proc(positions: []Vec3, indices: []u32) -> Mesh_Handle,
    draw_mesh    : proc(mesh: Mesh_Handle, color: Color),
    draw_line    : proc(a, b: Vec3, color: Color),
    // A unit sphere is built once per backend; this draws it with the given
    // center and radius so the logic layer never sees per-vertex data.
    draw_sphere  : proc(center: Vec3, radius: f32, color: Color),
    clear        : proc(color: Color),
    set_viewport : proc(w, h: int),
}

App :: struct {
    renderer:   Renderer,
    triangle:   Mesh_Handle,
    time:       f32,
}

initialize :: proc(app: ^App) {
    positions := []Vec3{
        {-0.6, -0.5, 0},
        { 0.6, -0.5, 0},
        { 0.0,  0.6, 0},
    }
    indices := []u32{0, 1, 2}
    app.triangle = app.renderer.create_mesh(positions, indices)
}

frame :: proc(app: ^App, dt: f32) {
    app.time += dt

    app.renderer.clear({0.10, 0.11, 0.15, 1.0})
    app.renderer.draw_mesh(app.triangle, {0.95, 0.45, 0.20, 1.0})

    // Cross-hair lines.
    app.renderer.draw_line({-0.9,  0.0, 0}, { 0.9,  0.0, 0}, {0.3, 0.7, 0.9, 1})
    app.renderer.draw_line({ 0.0, -0.9, 0}, { 0.0,  0.9, 0}, {0.3, 0.7, 0.9, 1})

    // Wandering sphere to exercise draw_sphere on both backends.
    cx := math.cos(app.time) * 0.5
    cy := math.sin(app.time) * 0.5
    app.renderer.draw_sphere({cx, cy, 0}, 0.15, {0.4, 0.95, 0.6, 1.0})
}

// ----------------------------------------------------------------------------
// Shared mesh generators.
// ----------------------------------------------------------------------------

// UV sphere. `stacks` is the number of latitude bands, `slices` the longitude.
// Returns positions for a unit sphere centred at the origin and the matching
// triangle index list. Caller owns the slices.
generate_uv_sphere :: proc(stacks, slices: int, allocator := context.allocator) -> (positions: []Vec3, indices: []u32) {
    context.allocator = allocator

    vert_count := (stacks + 1) * (slices + 1)
    positions = make([]Vec3, vert_count)

    i := 0
    for s := 0; s <= stacks; s += 1 {
        v     := f32(s) / f32(stacks)
        phi   := v * math.PI            // 0..π
        sin_p := math.sin(phi); cos_p := math.cos(phi)
        for sl := 0; sl <= slices; sl += 1 {
            u    := f32(sl) / f32(slices)
            theta := u * 2.0 * math.PI  // 0..2π
            sin_t := math.sin(theta); cos_t := math.cos(theta)
            positions[i] = Vec3{cos_t * sin_p, cos_p, sin_t * sin_p}
            i += 1
        }
    }

    indices = make([]u32, stacks * slices * 6)
    j := 0
    for s := 0; s < stacks; s += 1 {
        for sl := 0; sl < slices; sl += 1 {
            row0 := u32(s     * (slices + 1) + sl)
            row1 := u32((s+1) * (slices + 1) + sl)
            indices[j+0] = row0
            indices[j+1] = row1
            indices[j+2] = row0 + 1
            indices[j+3] = row0 + 1
            indices[j+4] = row1
            indices[j+5] = row1 + 1
            j += 6
        }
    }
    return
}
