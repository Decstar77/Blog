package geomesh_web

import "core:fmt"
import gl  "vendor:wasm/WebGL"
import glm "core:math/linalg/glsl"

import logic "../logic"

// WebGL2-flavored GLSL (ES 3.00). Same uniforms as the desktop shader.
VS_SRC :: `#version 300 es
layout(location = 0) in vec3 a_pos;
uniform mat4 u_view_proj;
uniform mat4 u_model;
void main() {
    gl_Position = u_view_proj * u_model * vec4(a_pos, 1.0);
}`

FS_SRC :: `#version 300 es
precision mediump float;
out vec4 frag_color;
uniform vec4 u_color;
void main() {
    frag_color = u_color;
}`

GL_Mesh :: struct {
    vao:         gl.VertexArrayObject,
    vbo, ibo:    gl.Buffer,
    index_count: int,
}

g_program       : gl.Program
g_u_color_loc   : i32
g_u_model_loc   : i32
g_u_vp_loc      : i32
g_meshes        : [dynamic]GL_Mesh
g_line_vao      : gl.VertexArrayObject
g_line_vbo      : gl.Buffer
g_tri_vao       : gl.VertexArrayObject
g_tri_vbo       : gl.Buffer
g_tri_ibo       : gl.Buffer

IDENTITY4 :: logic.Mat4{
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
}

renderer_init :: proc() {
    program, ok := gl.CreateProgramFromStrings({VS_SRC}, {FS_SRC})
    if !ok {
        fmt.eprintln("shader program failed to build")
        return
    }
    g_program     = program
    g_u_color_loc = gl.GetUniformLocation(program, "u_color")
    g_u_model_loc = gl.GetUniformLocation(program, "u_model")
    g_u_vp_loc    = gl.GetUniformLocation(program, "u_view_proj")

    gl.Enable(gl.DEPTH_TEST)

    g_line_vao = gl.CreateVertexArray()
    g_line_vbo = gl.CreateBuffer()
    gl.BindVertexArray(g_line_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_line_vbo)
    gl.BufferData(gl.ARRAY_BUFFER, 2 * size_of([3]f32), nil, gl.DYNAMIC_DRAW)
    gl.EnableVertexAttribArray(0)
    gl.VertexAttribPointer(0, 3, gl.FLOAT, false, size_of([3]f32), 0)
    gl.BindVertexArray(0)

    g_tri_vao = gl.CreateVertexArray()
    g_tri_vbo = gl.CreateBuffer()
    g_tri_ibo = gl.CreateBuffer()
    gl.BindVertexArray(g_tri_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_tri_vbo)
    gl.EnableVertexAttribArray(0)
    gl.VertexAttribPointer(0, 3, gl.FLOAT, false, size_of([3]f32), 0)
    gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, g_tri_ibo)
    gl.BindVertexArray(0)
}

make_renderer :: proc() -> logic.Renderer {
    return logic.Renderer{
        create_mesh             = gl_create_mesh,
        update_mesh             = gl_update_mesh,
        draw_mesh               = gl_draw_mesh,
        draw_mesh_xform         = gl_draw_mesh_xform,
        draw_triangles          = gl_draw_triangles,
        draw_line               = gl_draw_line,
        draw_line_overlay       = gl_draw_line_overlay,
        clear                   = gl_clear,
        set_viewport            = gl_set_viewport,
        set_view_projection     = gl_set_view_projection,
    }
}

@(private="file")
set_uniforms :: proc(color: logic.Color, model: logic.Mat4) {
    gl.UseProgram(g_program)
    c := []glm.vec4{ {color.r, color.g, color.b, color.a} }
    gl.Uniform4fv(g_u_color_loc, c)
    m := glm.mat4(model)
    gl.UniformMatrix4fv(g_u_model_loc, m)
}

@(private="file")
gl_create_mesh :: proc(positions: []logic.Vec3, indices: []u32) -> logic.Mesh_Handle {
    m: GL_Mesh
    m.index_count = len(indices)
    m.vao = gl.CreateVertexArray()
    m.vbo = gl.CreateBuffer()
    m.ibo = gl.CreateBuffer()

    gl.BindVertexArray(m.vao)

    gl.BindBuffer(gl.ARRAY_BUFFER, m.vbo)
    gl.BufferData(gl.ARRAY_BUFFER, len(positions) * size_of(logic.Vec3), raw_data(positions), gl.STATIC_DRAW)

    gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.ibo)
    gl.BufferData(gl.ELEMENT_ARRAY_BUFFER, len(indices) * size_of(u32), raw_data(indices), gl.STATIC_DRAW)

    gl.EnableVertexAttribArray(0)
    gl.VertexAttribPointer(0, 3, gl.FLOAT, false, size_of(logic.Vec3), 0)

    gl.BindVertexArray(0)

    append(&g_meshes, m)
    return logic.Mesh_Handle(len(g_meshes))
}

@(private="file")
gl_draw_mesh :: proc(mesh: logic.Mesh_Handle, color: logic.Color) {
    gl_draw_mesh_xform(mesh, IDENTITY4, color)
}

@(private="file")
gl_draw_mesh_xform :: proc(mesh: logic.Mesh_Handle, model: logic.Mat4, color: logic.Color) {
    idx := int(mesh) - 1
    if idx < 0 || idx >= len(g_meshes) do return
    m := g_meshes[idx]

    set_uniforms(color, model)
    gl.BindVertexArray(m.vao)
    gl.DrawElements(gl.TRIANGLES, m.index_count, gl.UNSIGNED_INT, nil)
    gl.BindVertexArray(0)
}

@(private="file")
gl_draw_triangles :: proc(positions: []logic.Vec3, indices: []u32, color: logic.Color) {
    if len(positions) == 0 || len(indices) == 0 do return
    set_uniforms(color, IDENTITY4)
    gl.BindVertexArray(g_tri_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_tri_vbo)
    gl.BufferData(gl.ARRAY_BUFFER, len(positions) * size_of(logic.Vec3), raw_data(positions), gl.DYNAMIC_DRAW)
    gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, g_tri_ibo)
    gl.BufferData(gl.ELEMENT_ARRAY_BUFFER, len(indices) * size_of(u32), raw_data(indices), gl.DYNAMIC_DRAW)
    gl.DrawElements(gl.TRIANGLES, len(indices), gl.UNSIGNED_INT, nil)
    gl.BindVertexArray(0)
}

@(private="file")
gl_update_mesh :: proc(mesh: logic.Mesh_Handle, positions: []logic.Vec3) {
    idx := int(mesh) - 1
    if idx < 0 || idx >= len(g_meshes) do return
    m := g_meshes[idx]
    gl.BindBuffer(gl.ARRAY_BUFFER, m.vbo)
    gl.BufferSubData(gl.ARRAY_BUFFER, 0, len(positions) * size_of(logic.Vec3), raw_data(positions))
}

@(private="file")
gl_draw_line_overlay :: proc(a, b: logic.Vec3, color: logic.Color) {
    verts := [2]logic.Vec3{a, b}
    set_uniforms(color, IDENTITY4)
    gl.Disable(gl.DEPTH_TEST)
    gl.BindVertexArray(g_line_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_line_vbo)
    gl.BufferSubData(gl.ARRAY_BUFFER, 0, size_of(verts), &verts)
    gl.DrawArrays(gl.LINES, 0, 2)
    gl.BindVertexArray(0)
    gl.Enable(gl.DEPTH_TEST)
}

@(private="file")
gl_draw_line :: proc(a, b: logic.Vec3, color: logic.Color) {
    verts := [2]logic.Vec3{a, b}

    set_uniforms(color, IDENTITY4)
    gl.BindVertexArray(g_line_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_line_vbo)
    gl.BufferSubData(gl.ARRAY_BUFFER, 0, size_of(verts), &verts)
    gl.DrawArrays(gl.LINES, 0, 2)
    gl.BindVertexArray(0)
}

@(private="file")
gl_clear :: proc(color: logic.Color) {
    gl.ClearColor(color.r, color.g, color.b, color.a)
    gl.Clear(u32(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT))
}

@(private="file")
gl_set_viewport :: proc(w, h: int) {
    gl.Viewport(0, 0, i32(w), i32(h))
}

@(private="file")
gl_set_view_projection :: proc(vp: logic.Mat4) {
    m := glm.mat4(vp)
    gl.UniformMatrix4fv(g_u_vp_loc, m)
}
