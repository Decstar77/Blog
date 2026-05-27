package geomesh_win32

import "core:fmt"
import gl  "vendor:OpenGL"

import logic "../logic"

// Vertex shader applies a per-draw model matrix then the camera
// view-projection. Most draws pass identity for u_model; mesh-instance draws
// (sphere/cylinder primitives) pass a real transform.
VS_SRC :: `#version 330 core
layout(location = 0) in vec3 a_pos;
uniform mat4 u_view_proj;
uniform mat4 u_model;
void main() {
    gl_Position = u_view_proj * u_model * vec4(a_pos, 1.0);
}`

FS_SRC :: `#version 330 core
out vec4 frag_color;
uniform vec4 u_color;
void main() {
    frag_color = u_color;
}`

// Separate line program: per-vertex color, no model matrix (lines are
// submitted in world space). Lets the frame's lines flush as one draw call.
LINE_VS_SRC :: `#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec4 a_color;
uniform mat4 u_view_proj;
out vec4 v_color;
void main() {
    v_color = a_color;
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
}`

LINE_FS_SRC :: `#version 330 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = v_color;
}`

GL_Mesh :: struct {
    vao, vbo, ibo: u32,
    index_count:   i32,
    primitive:     u32,
}

Line_Vertex :: struct {
    pos:   logic.Vec3,
    color: logic.Color,
}

g_program           : u32
g_u_color_loc       : i32
g_u_model_loc       : i32
g_u_vp_loc          : i32
g_meshes            : [dynamic]GL_Mesh
g_line_program      : u32
g_line_u_vp_loc     : i32
g_line_vao          : u32
g_line_vbo          : u32
g_line_vbo_capacity : int
g_lines             : [dynamic]Line_Vertex
g_tri_vao           : u32
g_tri_vbo           : u32
g_tri_ibo           : u32

IDENTITY4 :: logic.Mat4{
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
}

renderer_init :: proc() {
    // Report actual sample count -- the driver may grant fewer samples than
    // we requested in the GLFW hint.
    samples, sample_buffers: i32
    gl.GetIntegerv(gl.SAMPLES, &samples)
    gl.GetIntegerv(gl.SAMPLE_BUFFERS, &sample_buffers)
    fmt.println("GL_SAMPLES=", samples, " GL_SAMPLE_BUFFERS=", sample_buffers)

    program, ok := gl.load_shaders_source(VS_SRC, FS_SRC)
    if !ok {
        msg, _, link_msg, _ := gl.get_last_error_messages()
        fmt.eprintln("shader error:", msg, link_msg)
        return
    }
    g_program     = program
    g_u_color_loc = gl.GetUniformLocation(program, "u_color")
    g_u_model_loc = gl.GetUniformLocation(program, "u_model")
    g_u_vp_loc    = gl.GetUniformLocation(program, "u_view_proj")

    gl.Enable(gl.DEPTH_TEST)

    line_prog, line_ok := gl.load_shaders_source(LINE_VS_SRC, LINE_FS_SRC)
    if !line_ok {
        msg, _, link_msg, _ := gl.get_last_error_messages()
        fmt.eprintln("line shader error:", msg, link_msg)
        return
    }
    g_line_program  = line_prog
    g_line_u_vp_loc = gl.GetUniformLocation(line_prog, "u_view_proj")

    // Batched line VBO: filled by gl_draw_line, flushed once in gl_end_frame.
    gl.GenVertexArrays(1, &g_line_vao)
    gl.GenBuffers(1, &g_line_vbo)
    gl.BindVertexArray(g_line_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_line_vbo)
    gl.EnableVertexAttribArray(0)
    gl.VertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, size_of(Line_Vertex), 0)
    gl.EnableVertexAttribArray(1)
    gl.VertexAttribPointer(1, 4, gl.FLOAT, gl.FALSE, size_of(Line_Vertex), uintptr(size_of(logic.Vec3)))
    gl.BindVertexArray(0)

    // Dynamic VBO/IBO for transient triangle overlays.
    gl.GenVertexArrays(1, &g_tri_vao)
    gl.GenBuffers(1, &g_tri_vbo)
    gl.GenBuffers(1, &g_tri_ibo)
    gl.BindVertexArray(g_tri_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_tri_vbo)
    gl.EnableVertexAttribArray(0)
    gl.VertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, size_of([3]f32), 0)
    gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, g_tri_ibo)
    gl.BindVertexArray(0)

    text_renderer_init()
}

make_renderer :: proc() -> logic.Renderer {
    return logic.Renderer{
        create_mesh             = gl_create_mesh,
        update_mesh             = gl_update_mesh,
        draw_mesh               = gl_draw_mesh,
        draw_mesh_xform         = gl_draw_mesh_xform,
        draw_triangles          = gl_draw_triangles,
        draw_line               = gl_draw_line,
        end_frame               = gl_end_frame,
        clear                   = gl_clear,
        set_viewport            = gl_set_viewport,
        set_view_projection     = gl_set_view_projection,
        load_font               = gl_load_font,
        draw_text_3d            = gl_draw_text_3d,
    }
}

@(private="file")
set_uniforms :: proc(color: logic.Color, model: logic.Mat4) {
    gl.UseProgram(g_program)
    c := color
    gl.Uniform4fv(g_u_color_loc, 1, raw_data(c[:]))
    m := model
    gl.UniformMatrix4fv(g_u_model_loc, 1, gl.FALSE, &m[0, 0])
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
    gl_draw_mesh_xform(mesh, IDENTITY4, color)
}

@(private="file")
gl_draw_mesh_xform :: proc(mesh: logic.Mesh_Handle, model: logic.Mat4, color: logic.Color) {
    idx := int(mesh) - 1
    if idx < 0 || idx >= len(g_meshes) do return
    m := g_meshes[idx]

    set_uniforms(color, model)
    gl.BindVertexArray(m.vao)
    gl.DrawElements(m.primitive, m.index_count, gl.UNSIGNED_INT, nil)
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
    gl.DrawElements(gl.TRIANGLES, i32(len(indices)), gl.UNSIGNED_INT, nil)
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
gl_draw_line :: proc(a, b: logic.Vec3, color: logic.Color) {
    append(&g_lines, Line_Vertex{pos = a, color = color})
    append(&g_lines, Line_Vertex{pos = b, color = color})
}

@(private="file")
gl_end_frame :: proc() {
    if len(g_lines) == 0 do return

    bytes := len(g_lines) * size_of(Line_Vertex)
    gl.BindVertexArray(g_line_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_line_vbo)
    // Grow the VBO when needed; reuse with BufferSubData otherwise to avoid
    // re-allocating the storage every frame.
    if bytes > g_line_vbo_capacity {
        gl.BufferData(gl.ARRAY_BUFFER, bytes, raw_data(g_lines), gl.DYNAMIC_DRAW)
        g_line_vbo_capacity = bytes
    } else {
        gl.BufferSubData(gl.ARRAY_BUFFER, 0, bytes, raw_data(g_lines))
    }

    gl.UseProgram(g_line_program)
    gl.DrawArrays(gl.LINES, 0, i32(len(g_lines)))
    gl.BindVertexArray(0)

    clear(&g_lines)
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

@(private="file")
gl_set_view_projection :: proc(vp: logic.Mat4) {
    m := vp
    gl.UseProgram(g_program)
    gl.UniformMatrix4fv(g_u_vp_loc, 1, gl.FALSE, &m[0, 0])
    gl.UseProgram(g_line_program)
    gl.UniformMatrix4fv(g_line_u_vp_loc, 1, gl.FALSE, &m[0, 0])
    text_set_view_projection(vp)
}
