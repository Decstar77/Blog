package logic

import "core:math"
import "core:math/linalg"

Mesh_Handle :: distinct u32

INVALID_MESH :: Mesh_Handle(0)

Mat4 :: matrix[4, 4]f32

Renderer :: struct {
    create_mesh         : proc(positions: []Vec3, indices: []u32) -> Mesh_Handle,
    draw_mesh           : proc(mesh: Mesh_Handle, color: Color),
    draw_line           : proc(a, b: Vec3, color: Color),
    clear               : proc(color: Color),
    set_viewport        : proc(w, h: int),
    set_view_projection : proc(vp: Mat4),
}

// Per-frame input snapshot. Filled in by the platform layer (win32/web) and
// consumed by `update_camera` inside `frame`. All mouse deltas are in pixels
// since the previous frame; `scroll_dy` is accumulated wheel ticks.
Input :: struct {
    forward, back, left, right: bool,
    up, down:                   bool,
    boost:                      bool, // shift = sprint
    look_active:                bool, // RMB held
    mouse_dx, mouse_dy:         f32,
    scroll_dy:                  f32,
    aspect:                     f32,  // viewport w/h, used to refresh proj
}

// Unreal-Engine-style fly camera. Yaw/pitch are radians; yaw=0 looks down -Z,
// pitch=0 is the horizon, +pitch looks up.
Camera :: struct {
    position:   Vec3,
    yaw, pitch: f32,
    fov_y:      f32,
    aspect:     f32,
    near, far:  f32,
    move_speed: f32, // units per second; tuned by the scroll wheel
    look_speed: f32, // radians per pixel of mouse motion
}

App :: struct {
    renderer:   Renderer,
    plane:      HalfMesh,
    plane_mesh: Mesh_Handle,
    camera:     Camera,
    input:      Input,
    time:       f32,
}

initialize :: proc(app: ^App) {
    app.plane = create_plane(4, 4)

    positions, indices := halfmesh_to_triangles(&app.plane)
    defer delete(positions)
    defer delete(indices)
    app.plane_mesh = app.renderer.create_mesh(positions, indices)

    app.camera = Camera{
        position   = {0, 3, 6},
        yaw        = 0,
        pitch      = -0.45,
        fov_y      = math.PI * 0.33,
        aspect     = 16.0 / 9.0,
        near       = 0.05,
        far        = 500,
        move_speed = 5,
        look_speed = 0.0025,
    }
}

// Forward direction implied by yaw/pitch. yaw=0,pitch=0 -> -Z.
camera_forward :: proc(c: ^Camera) -> Vec3 {
    cp := math.cos(c.pitch); sp := math.sin(c.pitch)
    cy := math.cos(c.yaw);   sy := math.sin(c.yaw)
    return {sy * cp, sp, -cy * cp}
}

camera_right :: proc(c: ^Camera) -> Vec3 {
    cy := math.cos(c.yaw); sy := math.sin(c.yaw)
    return {cy, 0, sy}
}

camera_view_projection :: proc(c: ^Camera) -> Mat4 {
    fwd  := camera_forward(c)
    eye  := c.position
    view := linalg.matrix4_look_at_f32(eye, eye + fwd, {0, 1, 0})
    proj := linalg.matrix4_perspective_f32(c.fov_y, c.aspect, c.near, c.far)
    return proj * view
}

// Apply a frame of input to the camera. Movement only engages while RMB is
// held -- this matches Unreal's editor viewport. Scroll adjusts move speed
// geometrically (Unreal uses a similar log-style scaling).
update_camera :: proc(c: ^Camera, inp: Input, dt: f32) {
    if inp.aspect > 0 do c.aspect = inp.aspect

    if inp.scroll_dy != 0 {
        // Each notch multiplies/divides speed by ~1.2; clamp to a sane range.
        c.move_speed = clamp(c.move_speed * math.pow(f32(1.2), inp.scroll_dy), 0.1, 500)
    }

    if !inp.look_active do return

    c.yaw   += inp.mouse_dx * c.look_speed
    c.pitch -= inp.mouse_dy * c.look_speed
    limit := f32(math.PI * 0.49)
    c.pitch = clamp(c.pitch, -limit, limit)

    fwd   := camera_forward(c)
    right := camera_right(c)
    move  := Vec3{}
    if inp.forward do move += fwd
    if inp.back    do move -= fwd
    if inp.right   do move += right
    if inp.left    do move -= right
    if inp.up      do move += {0,  1, 0}
    if inp.down    do move += {0, -1, 0}

    if move != {0, 0, 0} {
        move = linalg.normalize(move)
        speed := c.move_speed * (inp.boost ? 4.0 : 1.0)
        c.position += move * speed * dt
    }
}

frame :: proc(app: ^App, dt: f32) {
    app.time += dt

    update_camera(&app.camera, app.input, dt)

    app.renderer.clear({0.10, 0.11, 0.15, 1.0})
    app.renderer.set_view_projection(camera_view_projection(&app.camera))

    // Filled plane.
    app.renderer.draw_mesh(app.plane_mesh, {0.45, 0.55, 0.85, 1.0})

    // Wireframe overlay so the half-edge structure is visible.
    edge_color := Color{1, 1, 1, 1}
    for e in app.plane.edges {
        if e.halfEdge == NONE do continue
        he := app.plane.halfedges[e.halfEdge]
        a  := app.plane.vertices[he.vert].position
        b  := app.plane.vertices[app.plane.halfedges[he.twin].vert].position
        app.renderer.draw_line(a, b, edge_color)
    }

    // World axes for orientation: X=red, Y=green, Z=blue.
    app.renderer.draw_line({0,0,0}, {1,0,0}, {0.95, 0.30, 0.30, 1})
    app.renderer.draw_line({0,0,0}, {0,1,0}, {0.30, 0.95, 0.30, 1})
    app.renderer.draw_line({0,0,0}, {0,0,1}, {0.30, 0.55, 0.95, 1})
}
