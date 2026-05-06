package logic

// Platform-agnostic application logic. The platform layer (win32 or web) is
// responsible for filling out the `Renderer` proc table on `App`, then calling
// `initialize` once and `frame` every tick.

Vec3  :: [3]f32
Color :: [4]f32

// Opaque mesh handle. Each platform decides what an integer means internally
// (an index into a slice of GL buffers / WebGL buffer objects).
Mesh_Handle :: distinct u32

INVALID_MESH :: Mesh_Handle(0)

Renderer :: struct {
    // Lifetime of `positions` / `indices` only needs to extend through the call.
    create_mesh  : proc(positions: []Vec3, indices: []u32) -> Mesh_Handle,
    draw_mesh    : proc(mesh: Mesh_Handle, color: Color),
    draw_line    : proc(a, b: Vec3, color: Color),
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

    // A couple of debug-style lines to exercise draw_line on both backends.
    app.renderer.draw_line({-0.9,  0.0, 0}, { 0.9,  0.0, 0}, {0.3, 0.7, 0.9, 1})
    app.renderer.draw_line({ 0.0, -0.9, 0}, { 0.0,  0.9, 0}, {0.3, 0.7, 0.9, 1})
}
