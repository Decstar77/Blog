package geomesh_win32

import "core:fmt"
import gl  "vendor:OpenGL"

import logic "../logic"

// Vertex shader applies a uniform translate (xyz) + uniform scale (w) before
// emitting the clip-space position. Default value (0,0,0,1) is a no-op.
VS_SRC :: `#version 330 core
layout(location = 0) in vec3 a_pos;
uniform vec4 u_transform; // xyz = translate, w = scale
void main() {
    gl_Position = vec4(a_pos * u_transform.w + u_transform.xyz, 1.0);
}`

FS_SRC :: `#version 330 core
out vec4 frag_color;
uniform vec4 u_color;
void main() {
    frag_color = u_color;
}`

GL_Mesh :: struct {
    vao, vbo, ibo: u32,
    index_count:   i32,
    primitive:     u32, // gl.TRIANGLES, gl.LINES, ...
}

g_program       : u32
g_u_color_loc   : i32
g_u_xform_loc   : i32
g_meshes        : [dynamic]GL_Mesh
g_line_vao      : u32
g_line_vbo      : u32
g_sphere_handle : logic.Mesh_Handle

SPHERE_STACKS :: 24
SPHERE_SLICES :: 32

renderer_init :: proc() {
    program, ok := gl.load_shaders_source(VS_SRC, FS_SRC)
    if !ok {
        msg, _, link_msg, _ := gl.get_last_error_messages()
        fmt.eprintln("shader error:", msg, link_msg)
        return
    }
    g_program = program
    g_u_color_loc = gl.GetUniformLocation(program, "u_color")
    g_u_xform_loc = gl.GetUniformLocation(program, "u_transform")

    // Dynamic VBO for line drawing (two vec3 vertices per call).
    gl.GenVertexArrays(1, &g_line_vao)
    gl.GenBuffers(1, &g_line_vbo)
    gl.BindVertexArray(g_line_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_line_vbo)
    gl.BufferData(gl.ARRAY_BUFFER, 2 * size_of([3]f32), nil, gl.DYNAMIC_DRAW)
    gl.EnableVertexAttribArray(0)
    gl.VertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, size_of([3]f32), 0)
    gl.BindVertexArray(0)

    // Build the unit sphere once and stash its handle for draw_sphere.
    sphere_pos, sphere_idx := logic.generate_uv_sphere(SPHERE_STACKS, SPHERE_SLICES)
    defer delete(sphere_pos)
    defer delete(sphere_idx)
    g_sphere_handle = gl_create_mesh(sphere_pos, sphere_idx)
}

make_renderer :: proc() -> logic.Renderer {
    return logic.Renderer{
        create_mesh  = gl_create_mesh,
        draw_mesh    = gl_draw_mesh,
        draw_line    = gl_draw_line,
        draw_sphere  = gl_draw_sphere,
        clear        = gl_clear,
        set_viewport = gl_set_viewport,
    }
}

@(private="file")
set_uniforms :: proc(color: logic.Color, translate: logic.Vec3, scale: f32) {
    gl.UseProgram(g_program)
    c := color
    gl.Uniform4fv(g_u_color_loc, 1, raw_data(c[:]))
    xform := [4]f32{translate.x, translate.y, translate.z, scale}
    gl.Uniform4fv(g_u_xform_loc, 1, raw_data(xform[:]))
}

@(private="file")
gl_create_mesh :: proc(positions: []logic.Vec3, indices: []u32) -> logic.Mesh_Handle {
    m: GL_Mesh
    m.index_count = i32(len(indices))
    m.primitive   = gl.TRIANGLES

    gl.GenVertexArrays(1, &m.vao)
    gl.GenBuffers(1, &m.vbo)
    gl.GenBuffers(1, &m.ibo)

    gl.BindVertexArray(m.vao)

    gl.BindBuffer(gl.ARRAY_BUFFER, m.vbo)
    gl.BufferData(gl.ARRAY_BUFFER, len(positions) * size_of(logic.Vec3), raw_data(positions), gl.STATIC_DRAW)

    gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.ibo)
    gl.BufferData(gl.ELEMENT_ARRAY_BUFFER, len(indices) * size_of(u32), raw_data(indices), gl.STATIC_DRAW)

    gl.EnableVertexAttribArray(0)
    gl.VertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, size_of(logic.Vec3), 0)

    gl.BindVertexArray(0)

    append(&g_meshes, m)
    return logic.Mesh_Handle(len(g_meshes)) // 1-based; 0 reserved for INVALID_MESH
}

@(private="file")
gl_draw_mesh :: proc(mesh: logic.Mesh_Handle, color: logic.Color) {
    idx := int(mesh) - 1
    if idx < 0 || idx >= len(g_meshes) do return
    m := g_meshes[idx]

    set_uniforms(color, {0, 0, 0}, 1)
    gl.BindVertexArray(m.vao)
    gl.DrawElements(m.primitive, m.index_count, gl.UNSIGNED_INT, nil)
    gl.BindVertexArray(0)
}

@(private="file")
gl_draw_line :: proc(a, b: logic.Vec3, color: logic.Color) {
    verts := [2]logic.Vec3{a, b}

    set_uniforms(color, {0, 0, 0}, 1)
    gl.BindVertexArray(g_line_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_line_vbo)
    gl.BufferSubData(gl.ARRAY_BUFFER, 0, size_of(verts), &verts)
    gl.DrawArrays(gl.LINES, 0, 2)
    gl.BindVertexArray(0)
}

@(private="file")
gl_draw_sphere :: proc(center: logic.Vec3, radius: f32, color: logic.Color) {
    idx := int(g_sphere_handle) - 1
    if idx < 0 || idx >= len(g_meshes) do return
    m := g_meshes[idx]

    set_uniforms(color, center, radius)
    gl.BindVertexArray(m.vao)
    gl.DrawElements(m.primitive, m.index_count, gl.UNSIGNED_INT, nil)
    gl.BindVertexArray(0)
}

@(private="file")
gl_clear :: proc(color: logic.Color) {
    gl.ClearColor(color.r, color.g, color.b, color.a)
    gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
}

@(private="file")
gl_set_viewport :: proc(w, h: int) {
    gl.Viewport(0, 0, i32(w), i32(h))
}
