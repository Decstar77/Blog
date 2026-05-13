package geomesh_win32

// MSDF text renderer. Each font is an atlas PNG (produced by msdf-atlas-gen,
// `-type msdf`) + a JSON describing per-glyph plane/atlas bounds. The shader
// reconstructs distance from the RGB median and antialiases using screen-space
// derivatives of the UV, so glyphs stay crisp at any on-screen size.

import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:strings"

import gl    "vendor:OpenGL"
import stbi  "vendor:stb/image"

import logic "../logic"

TEXT_VS :: `#version 330 core
layout(location = 0) in vec3 a_anchor;   // world-space anchor (em origin)
layout(location = 1) in vec2 a_offset;   // em-space offset from anchor (x right, y up)
layout(location = 2) in vec2 a_uv;
out vec2 v_uv;
uniform mat4 u_view_proj;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
void main() {
    vec3 world = a_anchor + a_offset.x * u_cam_right + a_offset.y * u_cam_up;
    gl_Position = u_view_proj * vec4(world, 1.0);
    v_uv = a_uv;
}`

TEXT_FS :: `#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_msdf;
uniform vec4  u_color;
uniform float u_px_range;
float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}
// Convert glyph-space distance into screen pixels via UV derivatives.
float screen_px_range() {
    vec2 unit_range    = vec2(u_px_range) / vec2(textureSize(u_msdf, 0));
    vec2 screen_tex_sz = vec2(1.0) / fwidth(v_uv);
    return max(0.5, dot(unit_range, screen_tex_sz));
}
void main() {
    vec3  msd  = texture(u_msdf, v_uv).rgb;
    float sd   = median(msd.r, msd.g, msd.b);
    float dist = screen_px_range() * (sd - 0.5);
    float a    = clamp(dist + 0.5, 0.0, 1.0);
    if (a < 0.001) discard;
    frag = vec4(u_color.rgb, u_color.a * a);
}`

// ---- JSON schema mirrors msdf-atlas-gen's output. -------------------------

@(private="file")
JSON_Atlas :: struct {
    type:          string `json:"type"`,
    distanceRange: f32    `json:"distanceRange"`,
    size:          f32    `json:"size"`,
    width:         f32    `json:"width"`,
    height:        f32    `json:"height"`,
    yOrigin:       string `json:"yOrigin"`,
}

@(private="file")
JSON_Metrics :: struct {
    emSize:             f32 `json:"emSize"`,
    lineHeight:         f32 `json:"lineHeight"`,
    ascender:           f32 `json:"ascender"`,
    descender:          f32 `json:"descender"`,
    underlineY:         f32 `json:"underlineY"`,
    underlineThickness: f32 `json:"underlineThickness"`,
}

@(private="file")
JSON_Bounds :: struct {
    left:   f32 `json:"left"`,
    bottom: f32 `json:"bottom"`,
    right:  f32 `json:"right"`,
    top:    f32 `json:"top"`,
}

@(private="file")
JSON_Glyph :: struct {
    unicode:     u32         `json:"unicode"`,
    advance:     f32         `json:"advance"`,
    planeBounds: JSON_Bounds `json:"planeBounds"`,
    atlasBounds: JSON_Bounds `json:"atlasBounds"`,
}

@(private="file")
JSON_Font :: struct {
    atlas:   JSON_Atlas    `json:"atlas"`,
    metrics: JSON_Metrics  `json:"metrics"`,
    glyphs:  []JSON_Glyph  `json:"glyphs"`,
}

// ---- runtime representation -----------------------------------------------

Glyph :: struct {
    advance:                  f32,
    // em-space quad corners (x right, y up; y can be negative for descenders).
    p_left, p_bottom, p_right, p_top: f32,
    // UV (0..1) into the atlas.
    u_left, u_bottom, u_right, u_top: f32,
}

GL_Font :: struct {
    texture:     u32,
    px_range:    f32,
    line_height: f32,
    glyphs:      map[rune]Glyph,
}

g_fonts:        [dynamic]GL_Font
g_text_program: u32
g_text_loc:     struct {
    view_proj: i32,
    cam_right: i32,
    cam_up:    i32,
    color:     i32,
    px_range:  i32,
    msdf:      i32,
}
g_text_vao: u32
g_text_vbo: u32

// Per-vertex layout: vec3 anchor, vec2 offset_em, vec2 uv -> 7 floats.
@(private="file")
TEXT_VTX_FLOATS :: 7
@(private="file")
TEXT_MAX_QUADS  :: 1024

text_renderer_init :: proc() {
    program, ok := gl.load_shaders_source(TEXT_VS, TEXT_FS)
    if !ok {
        msg, _, link_msg, _ := gl.get_last_error_messages()
        fmt.eprintln("text shader error:", msg, link_msg)
        return
    }
    g_text_program = program
    g_text_loc.view_proj = gl.GetUniformLocation(program, "u_view_proj")
    g_text_loc.cam_right = gl.GetUniformLocation(program, "u_cam_right")
    g_text_loc.cam_up    = gl.GetUniformLocation(program, "u_cam_up")
    g_text_loc.color     = gl.GetUniformLocation(program, "u_color")
    g_text_loc.px_range  = gl.GetUniformLocation(program, "u_px_range")
    g_text_loc.msdf      = gl.GetUniformLocation(program, "u_msdf")

    gl.GenVertexArrays(1, &g_text_vao)
    gl.GenBuffers(1, &g_text_vbo)
    gl.BindVertexArray(g_text_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_text_vbo)
    stride := i32(TEXT_VTX_FLOATS * size_of(f32))
    gl.BufferData(gl.ARRAY_BUFFER, int(stride) * 6 * TEXT_MAX_QUADS, nil, gl.DYNAMIC_DRAW)
    gl.EnableVertexAttribArray(0)
    gl.VertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, stride, 0)
    gl.EnableVertexAttribArray(1)
    gl.VertexAttribPointer(1, 2, gl.FLOAT, gl.FALSE, stride, 3 * size_of(f32))
    gl.EnableVertexAttribArray(2)
    gl.VertexAttribPointer(2, 2, gl.FLOAT, gl.FALSE, stride, 5 * size_of(f32))
    gl.BindVertexArray(0)
}

gl_load_font :: proc(png_path, json_path: string) -> logic.Font_Handle {
    // -- JSON --------------------------------------------------------------
    json_bytes, json_ok := os.read_entire_file(json_path)
    if !json_ok {
        fmt.eprintln("font: failed to read", json_path)
        return logic.INVALID_FONT
    }
    defer delete(json_bytes)

    parsed: JSON_Font
    if err := json.unmarshal(json_bytes, &parsed); err != nil {
        fmt.eprintln("font: json parse failed for", json_path, "->", err)
        return logic.INVALID_FONT
    }

    // -- PNG ---------------------------------------------------------------
    stbi.set_flip_vertically_on_load(1) // MSDF JSON uses bottom-left origin
    cpath := strings.clone_to_cstring(png_path); defer delete(cpath)
    w, h, channels: i32
    pixels := stbi.load(cpath, &w, &h, &channels, 4)
    if pixels == nil {
        fmt.eprintln("font: failed to load", png_path)
        return logic.INVALID_FONT
    }
    defer stbi.image_free(pixels)

    tex: u32
    gl.GenTextures(1, &tex)
    gl.BindTexture(gl.TEXTURE_2D, tex)
    gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

    // -- build glyph map ---------------------------------------------------
    aw := parsed.atlas.width
    ah := parsed.atlas.height
    f := GL_Font{
        texture     = tex,
        px_range    = parsed.atlas.distanceRange,
        line_height = parsed.metrics.lineHeight,
        glyphs      = make(map[rune]Glyph, len(parsed.glyphs)),
    }
    for g in parsed.glyphs {
        gly := Glyph{
            advance  = g.advance,
            p_left   = g.planeBounds.left,
            p_bottom = g.planeBounds.bottom,
            p_right  = g.planeBounds.right,
            p_top    = g.planeBounds.top,
            // Atlas pixel bounds -> UV (atlas y is bottom-origin, so is the
            // texture after the flip-on-load above -> straight divide).
            u_left   = g.atlasBounds.left   / aw,
            u_bottom = g.atlasBounds.bottom / ah,
            u_right  = g.atlasBounds.right  / aw,
            u_top    = g.atlasBounds.top    / ah,
        }
        f.glyphs[rune(g.unicode)] = gly
    }

    append(&g_fonts, f)
    return logic.Font_Handle(len(g_fonts)) // 1-based
}

gl_draw_text_3d :: proc(handle: logic.Font_Handle, text: string, anchor: logic.Vec3,
                        em_size: f32, color: logic.Color,
                        cam_right, cam_up: logic.Vec3) {
    idx := int(handle) - 1
    if idx < 0 || idx >= len(g_fonts) || len(text) == 0 do return
    font := &g_fonts[idx]

    // Center the string horizontally so labels sit over their anchor point.
    // (Vertical: baseline at anchor.)
    total_advance: f32 = 0
    for r in text {
        if g, ok := font.glyphs[r]; ok do total_advance += g.advance
    }
    pen_x := -total_advance * 0.5

    // Build interleaved vertex buffer for all quads.
    verts: [dynamic]f32
    defer delete(verts)
    reserve(&verts, len(text) * 6 * TEXT_VTX_FLOATS)

    R := cam_right * em_size
    U := cam_up    * em_size
    _ = R; _ = U // shader expands; we only need em-space offsets here.

    push_vert :: proc(verts: ^[dynamic]f32, anchor: logic.Vec3, ox, oy, u, v: f32) {
        append(verts, anchor.x, anchor.y, anchor.z, ox, oy, u, v)
    }

    for r in text {
        g, ok := font.glyphs[r]
        if !ok { pen_x += 0.6; continue }
        if g.p_right > g.p_left { // skip space (no quad)
            l := (pen_x + g.p_left)   * em_size
            b := (g.p_bottom)         * em_size
            rg := (pen_x + g.p_right) * em_size
            t := (g.p_top)            * em_size
            // Two triangles (BL, BR, TR) and (BL, TR, TL).
            push_vert(&verts, anchor, l,  b, g.u_left,  g.u_bottom)
            push_vert(&verts, anchor, rg, b, g.u_right, g.u_bottom)
            push_vert(&verts, anchor, rg, t, g.u_right, g.u_top)
            push_vert(&verts, anchor, l,  b, g.u_left,  g.u_bottom)
            push_vert(&verts, anchor, rg, t, g.u_right, g.u_top)
            push_vert(&verts, anchor, l,  t, g.u_left,  g.u_top)
        }
        pen_x += g.advance
    }

    quad_count := len(verts) / (TEXT_VTX_FLOATS * 6)
    if quad_count == 0 do return

    // -- draw --------------------------------------------------------------
    gl.UseProgram(g_text_program)
    gl.Uniform3f(g_text_loc.cam_right, cam_right.x, cam_right.y, cam_right.z)
    gl.Uniform3f(g_text_loc.cam_up,    cam_up.x,    cam_up.y,    cam_up.z)
    c := color
    gl.Uniform4fv(g_text_loc.color, 1, raw_data(c[:]))
    gl.Uniform1f(g_text_loc.px_range, font.px_range)
    gl.Uniform1i(g_text_loc.msdf, 0)

    gl.ActiveTexture(gl.TEXTURE0)
    gl.BindTexture(gl.TEXTURE_2D, font.texture)

    // Blending on, depth write off: labels compose with the scene but don't
    // occlude other transparent geometry.
    gl.Enable(gl.BLEND)
    gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
    gl.DepthMask(gl.FALSE)

    gl.BindVertexArray(g_text_vao)
    gl.BindBuffer(gl.ARRAY_BUFFER, g_text_vbo)
    gl.BufferSubData(gl.ARRAY_BUFFER, 0, len(verts) * size_of(f32), raw_data(verts))
    gl.DrawArrays(gl.TRIANGLES, 0, i32(quad_count * 6))
    gl.BindVertexArray(0)

    gl.DepthMask(gl.TRUE)
}

// View-projection comes through the shared uniform name; set once per frame.
text_set_view_projection :: proc(vp: logic.Mat4) {
    if g_text_program == 0 do return
    m := vp
    gl.UseProgram(g_text_program)
    gl.UniformMatrix4fv(g_text_loc.view_proj, 1, gl.FALSE, &m[0, 0])
}
