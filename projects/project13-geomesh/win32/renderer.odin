package geomesh_win32

import "core:fmt"
import gl  "vendor:OpenGL"

import logic "../logic"

VS_SRC :: `#version 330 core
layout(location = 0) in vec3 a_pos;
void main() {
    gl_Position = vec4(a_pos, 1.0);
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
}

g_program     : u32
g_u_color_loc : i32
g_meshes      : [dynamic]GL_Mesh
g_line_vao    : u32
g_line_vbo    : u32

renderer_init :: proc() {
    program, ok := gl.load_shaders_source(VS_SRC, FS_SRC)
    if !ok {
        msg, _, link_msg, _ := gl.get_last_error_messages()
        fmt.eprintln("shader error:", msg, link_msg)
        return
    }
    g_program = program
    g_u_color_loc = gl.GetUniformLocation(program, "u_color")

    // Dedicated dynamic VBO for line drawing (two vec3 vertices per call).
    gl.GenVertexArrays(1, &g_line_vao)
    gl.GenBuffers(1, &g_line_vbo)
    gl.BindVertexArray(g_line_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_line_vbo)
    gl.BufferData(gl.ARRAY_BUFFER, 2 * size_of([3]f32), nil, gl.DYNAMIC_DRAW)
    gl.EnableVertexAttribArray(0)
    gl.VertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, size_of([3]f32), 0)
    gl.BindVertexArray(0)
}

// Hook for batched submission. We currently submit immediately, so this is a no-op.
renderer_flush :: proc() {}

make_renderer :: proc() -> logic.Renderer {
    return logic.Renderer{
        create_mesh  = gl_create_mesh,
        draw_mesh    = gl_draw_mesh,
        draw_line    = gl_draw_line,
        clear        = gl_clear,
        set_viewport = gl_set_viewport,
    }
}

@(private="file")
gl_create_mesh :: proc(positions: []logic.Vec3, indices: []u32) -> logic.Mesh_Handle {
    m: GL_Mesh
    m.index_count = i32(len(indices))

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

    gl.UseProgram(g_program)
    c := color
    gl.Uniform4fv(g_u_color_loc, 1, raw_data(c[:]))
    gl.BindVertexArray(m.vao)
    gl.DrawElements(gl.TRIANGLES, m.index_count, gl.UNSIGNED_INT, nil)
    gl.BindVertexArray(0)
}

@(private="file")
gl_draw_line :: proc(a, b: logic.Vec3, color: logic.Color) {
    verts := [2]logic.Vec3{a, b}

    gl.UseProgram(g_program)
    c := color
    gl.Uniform4fv(g_u_color_loc, 1, raw_data(c[:]))

    gl.BindVertexArray(g_line_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_line_vbo)
    gl.BufferSubData(gl.ARRAY_BUFFER, 0, size_of(verts), &verts)
    gl.DrawArrays(gl.LINES, 0, 2)
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
