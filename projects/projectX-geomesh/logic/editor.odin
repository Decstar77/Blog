package logic

// Selection + translation-gizmo state and the math that drives them.
// All renderer-agnostic; the platform layer just funnels mouse/keys through
// `Input` and the editor decides what to pick, drag, and translate.

import "core:math"
import "core:math/linalg"

Selection_Mode :: enum { Vertex, Edge, Face }

Selection :: struct {
    has:   bool,
    mode:  Selection_Mode,
    index: u32, // vertex / edge / face index, depending on mode
}

// Active translation drag. `axis` is 0=X, 1=Y, 2=Z. `last_t` is the parameter
// along the world-aligned axis line (passing through the drag-start centroid)
// where the mouse ray was closest on the previous frame -- the per-frame delta
// is just `(t_now - last_t) * axis_dir`, applied to the selection.
Gizmo :: struct {
    dragging: bool,
    axis:     int,
    origin:   Vec3, // axis line passes through this point (frozen at drag start)
    last_t:   f32,
    hover_axis: int, // -1 if none; updated every frame for highlighting
}

Editor :: struct {
    mode:      Selection_Mode,
    selection: Selection,
    gizmo:     Gizmo,
    // Edge-detection for one-shot inputs (LMB click, mode keys).
    prev_lmb:  bool,
    prev_k1:   bool,
    prev_k2:   bool,
    prev_k3:   bool,
}

// Visual scaling so the gizmo holds roughly the same on-screen size regardless
// of how far the camera is from the selection.
GIZMO_SCREEN_FRACTION :: f32(0.15)
AXIS_PICK_PIXELS      :: f32(10)
ELEMENT_PICK_PIXELS   :: f32(14)

// ---------------------------------------------------------------- selection --

selection_clear :: proc(s: ^Selection) {
    s.has = false
}

// Centroid of the currently-selected element, used as the gizmo origin.
selection_centroid :: proc(m: ^HalfMesh, s: Selection) -> (Vec3, bool) {
    if !s.has do return {}, false
    switch s.mode {
    case .Vertex:
        if int(s.index) >= len(m.vertices) do return {}, false
        return m.vertices[s.index].position, true
    case .Edge:
        if int(s.index) >= len(m.edges) do return {}, false
        e := m.edges[s.index]
        if e.halfEdge == NONE do return {}, false
        he := m.halfedges[e.halfEdge]
        a  := m.vertices[he.vert].position
        b  := m.vertices[m.halfedges[he.twin].vert].position
        return (a + b) * 0.5, true
    case .Face:
        if int(s.index) >= len(m.faces) do return {}, false
        f := m.faces[s.index]
        if f.halfEdge == NONE do return {}, false
        sum := Vec3{}
        n   := f32(0)
        h   := f.halfEdge
        for {
            sum += m.vertices[m.halfedges[h].vert].position
            n   += 1
            h = m.halfedges[h].next
            if h == f.halfEdge do break
        }
        return sum / n, true
    }
    return {}, false
}

// Walk the half-edge loop of a face and return its (de-duplicated) vertices.
@(private="file")
face_vertex_indices :: proc(m: ^HalfMesh, face_idx: u32, out: ^[dynamic]u32) {
    clear(out)
    f := m.faces[face_idx]
    if f.halfEdge == NONE do return
    h := f.halfEdge
    for {
        v := m.halfedges[h].vert
        already := false
        for x in out do if x == v { already = true; break }
        if !already do append(out, v)
        h = m.halfedges[h].next
        if h == f.halfEdge do break
    }
}

// Translate every vertex implicated by the current selection.
selection_translate :: proc(m: ^HalfMesh, s: Selection, delta: Vec3) {
    if !s.has do return
    switch s.mode {
    case .Vertex:
        if int(s.index) < len(m.vertices) do m.vertices[s.index].position += delta
    case .Edge:
        if int(s.index) >= len(m.edges) do return
        e := m.edges[s.index]
        if e.halfEdge == NONE do return
        he := m.halfedges[e.halfEdge]
        m.vertices[he.vert].position += delta
        m.vertices[m.halfedges[he.twin].vert].position += delta
    case .Face:
        if int(s.index) >= len(m.faces) do return
        verts: [dynamic]u32
        defer delete(verts)
        face_vertex_indices(m, s.index, &verts)
        for v in verts do m.vertices[v].position += delta
    }
}

// --------------------------------------------------------------------- math --

// Mouse position -> world-space ray (origin on near plane, normalized dir).
make_pick_ray :: proc(c: ^Camera, mouse_x, mouse_y, vw, vh: f32) -> (origin, dir: Vec3) {
    vp  := camera_view_projection(c)
    inv := linalg.matrix4_inverse(vp)
    ndc_x := (mouse_x / vw) * 2 - 1
    ndc_y := 1 - (mouse_y / vh) * 2
    near := inv * [4]f32{ndc_x, ndc_y, -1, 1}
    far  := inv * [4]f32{ndc_x, ndc_y,  1, 1}
    np := Vec3{near.x, near.y, near.z} / near.w
    fp := Vec3{far.x,  far.y,  far.z}  / far.w
    return np, linalg.normalize(fp - np)
}

// World point -> pixel coordinates (top-left origin). `behind` is true if the
// point is on or behind the near plane and should be skipped.
project_point :: proc(vp: Mat4, p: Vec3, vw, vh: f32) -> (sx, sy: f32, behind: bool) {
    v := vp * [4]f32{p.x, p.y, p.z, 1}
    if v.w <= 0.0001 do return 0, 0, true
    sx = ((v.x / v.w) * 0.5 + 0.5) * vw
    sy = (1 - ((v.y / v.w) * 0.5 + 0.5)) * vh
    return sx, sy, false
}

// Pixel distance from `p` to the segment (a, b). Clamped to the segment ends.
@(private="file")
dist_point_segment_2d :: proc(px, py, ax, ay, bx, by: f32) -> f32 {
    dx := bx - ax
    dy := by - ay
    L2 := dx*dx + dy*dy
    t  := f32(0)
    if L2 > 0 do t = clamp(((px-ax)*dx + (py-ay)*dy) / L2, 0, 1)
    cx := ax + t*dx
    cy := ay + t*dy
    return math.sqrt((px-cx)*(px-cx) + (py-cy)*(py-cy))
}

// Closest parameters on two infinite 3D lines. Used to drag the gizmo:
// p1 = o1 + t1*d1 (mouse ray), p2 = o2 + t2*d2 (axis line).
@(private="file")
closest_t_on_line2 :: proc(o1, d1, o2, d2: Vec3) -> f32 {
    r := o1 - o2
    a := linalg.dot(d1, d1)
    b := linalg.dot(d1, d2)
    c := linalg.dot(d2, d2)
    d := linalg.dot(d1, r)
    e := linalg.dot(d2, r)
    denom := a*c - b*b
    if abs(denom) < 1e-6 do return e / c
    return (a*e - b*d) / denom
}

// Möller-Trumbore. Returns the ray parameter `t` if the front side is hit.
@(private="file")
ray_triangle :: proc(o, d, a, b, c: Vec3) -> (t: f32, hit: bool) {
    e1 := b - a
    e2 := c - a
    p  := linalg.cross(d, e2)
    det := linalg.dot(e1, p)
    if abs(det) < 1e-7 do return
    inv := 1.0 / det
    s := o - a
    u := linalg.dot(s, p) * inv
    if u < 0 || u > 1 do return
    q := linalg.cross(s, e1)
    v := linalg.dot(d, q) * inv
    if v < 0 || u + v > 1 do return
    t  = linalg.dot(e2, q) * inv
    hit = t > 0
    return
}

// ----------------------------------------------------------------- picking --

// Best vertex/edge/face under the cursor for the active mode. `found` is false
// if nothing is within the screen-space pick threshold.
pick_element :: proc(m: ^HalfMesh, mode: Selection_Mode, c: ^Camera,
                     mouse_x, mouse_y, vw, vh: f32) -> (idx: u32, found: bool) {
    vp := camera_view_projection(c)

    switch mode {
    case .Vertex:
        best  := f32(ELEMENT_PICK_PIXELS)
        for v, i in m.vertices {
            sx, sy, behind := project_point(vp, v.position, vw, vh)
            if behind do continue
            d := math.sqrt((sx-mouse_x)*(sx-mouse_x) + (sy-mouse_y)*(sy-mouse_y))
            if d < best { best = d; idx = u32(i); found = true }
        }
    case .Edge:
        best := f32(ELEMENT_PICK_PIXELS)
        for e, i in m.edges {
            if e.halfEdge == NONE do continue
            he := m.halfedges[e.halfEdge]
            a  := m.vertices[he.vert].position
            b  := m.vertices[m.halfedges[he.twin].vert].position
            ax, ay, ab := project_point(vp, a, vw, vh)
            bx, by, bb := project_point(vp, b, vw, vh)
            if ab || bb do continue
            d := dist_point_segment_2d(mouse_x, mouse_y, ax, ay, bx, by)
            if d < best { best = d; idx = u32(i); found = true }
        }
    case .Face:
        origin, dir := make_pick_ray(c, mouse_x, mouse_y, vw, vh)
        best := f32(1e30)
        for f, i in m.faces {
            if f.halfEdge == NONE do continue
            // Triangulate as a fan from the anchor vertex (matches
            // halfmesh_to_triangles), test each triangle.
            anchor_he := f.halfEdge
            anchor    := m.vertices[m.halfedges[anchor_he].vert].position
            h := m.halfedges[anchor_he].next
            for {
                next := m.halfedges[h].next
                if next == anchor_he do break
                pb := m.vertices[m.halfedges[h].vert].position
                pc := m.vertices[m.halfedges[next].vert].position
                if t, hit := ray_triangle(origin, dir, anchor, pb, pc); hit && t < best {
                    best = t; idx = u32(i); found = true
                }
                h = next
            }
        }
    }
    return
}

// ------------------------------------------------------------------ gizmo --

axis_dir :: proc(axis: int) -> Vec3 {
    switch axis {
    case 0: return {1, 0, 0}
    case 1: return {0, 1, 0}
    case 2: return {0, 0, 1}
    }
    return {0, 0, 0}
}

// On-screen size; matches the convention used by every common DCC editor.
gizmo_size :: proc(c: ^Camera, origin: Vec3) -> f32 {
    d := linalg.length(origin - c.position)
    return max(d * GIZMO_SCREEN_FRACTION, 0.05)
}

// Closest axis to the cursor in screen space. Returns -1 when no axis is
// within `AXIS_PICK_PIXELS`.
pick_gizmo_axis :: proc(c: ^Camera, origin: Vec3, mouse_x, mouse_y, vw, vh: f32) -> int {
    vp   := camera_view_projection(c)
    size := gizmo_size(c, origin)
    ox, oy, ob := project_point(vp, origin, vw, vh)
    if ob do return -1
    best := f32(AXIS_PICK_PIXELS)
    hit  := -1
    for a in 0..<3 {
        tip := origin + axis_dir(a) * size
        tx, ty, tb := project_point(vp, tip, vw, vh)
        if tb do continue
        d := dist_point_segment_2d(mouse_x, mouse_y, ox, oy, tx, ty)
        if d < best { best = d; hit = a }
    }
    return hit
}

// ----------------------------------------------------------- main entrypoint --

// Run one editor tick. Returns true if mesh geometry was modified (so the
// caller can re-upload the GPU mesh).
editor_update :: proc(ed: ^Editor, mesh: ^HalfMesh, c: ^Camera, inp: Input) -> (mesh_dirty: bool) {
    // Mode switching (1=vertex, 2=edge, 3=face). Edge-triggered.
    switch_mode :: proc(ed: ^Editor, m: Selection_Mode) {
        if ed.mode != m {
            ed.mode = m
            selection_clear(&ed.selection)
            ed.gizmo.dragging = false
        }
    }
    if inp.k1 && !ed.prev_k1 do switch_mode(ed, .Vertex)
    if inp.k2 && !ed.prev_k2 do switch_mode(ed, .Edge)
    if inp.k3 && !ed.prev_k3 do switch_mode(ed, .Face)
    ed.prev_k1 = inp.k1
    ed.prev_k2 = inp.k2
    ed.prev_k3 = inp.k3

    vw := inp.viewport_w
    vh := inp.viewport_h
    // Picking + dragging is gated on a sane viewport and on NOT being in
    // camera-look mode (RMB held), so the two interactions never conflict.
    interact := vw > 0 && vh > 0 && !inp.look_active

    // Hover highlight for the gizmo (purely cosmetic, but cheap).
    ed.gizmo.hover_axis = -1
    if interact && ed.selection.has && !ed.gizmo.dragging {
        if origin, ok := selection_centroid(mesh, ed.selection); ok {
            ed.gizmo.hover_axis = pick_gizmo_axis(c, origin, inp.mouse_x, inp.mouse_y, vw, vh)
        }
    }

    lmb_pressed  := interact && inp.lmb && !ed.prev_lmb
    lmb_released := !inp.lmb && ed.prev_lmb

    // -- start drag ----------------------------------------------------------
    if lmb_pressed && ed.selection.has {
        if origin, ok := selection_centroid(mesh, ed.selection); ok {
            axis := pick_gizmo_axis(c, origin, inp.mouse_x, inp.mouse_y, vw, vh)
            if axis >= 0 {
                ro, rd := make_pick_ray(c, inp.mouse_x, inp.mouse_y, vw, vh)
                ed.gizmo.dragging = true
                ed.gizmo.axis     = axis
                ed.gizmo.origin   = origin
                ed.gizmo.last_t   = closest_t_on_line2(ro, rd, origin, axis_dir(axis))
            }
        }
    }

    // -- continue drag -------------------------------------------------------
    if ed.gizmo.dragging && inp.lmb && interact {
        ro, rd := make_pick_ray(c, inp.mouse_x, inp.mouse_y, vw, vh)
        t      := closest_t_on_line2(ro, rd, ed.gizmo.origin, axis_dir(ed.gizmo.axis))
        delta  := (t - ed.gizmo.last_t) * axis_dir(ed.gizmo.axis)
        if delta != {0, 0, 0} {
            selection_translate(mesh, ed.selection, delta)
            mesh_dirty = true
        }
        ed.gizmo.last_t = t
    }

    // -- pick (only fires on a click that didn't start a drag) ---------------
    if lmb_pressed && !ed.gizmo.dragging {
        if idx, ok := pick_element(mesh, ed.mode, c, inp.mouse_x, inp.mouse_y, vw, vh); ok {
            ed.selection = Selection{has = true, mode = ed.mode, index = idx}
        } else {
            selection_clear(&ed.selection)
        }
    }

    if lmb_released do ed.gizmo.dragging = false
    ed.prev_lmb = inp.lmb
    return
}

// ---------------------------------------------------------------- rendering --

// Highlight the active selection. Drawn as overlay (no depth test) so it
// always sits on top of the underlying mesh wireframe.
draw_selection_highlight :: proc(r: ^Renderer, m: ^HalfMesh, s: Selection, c: ^Camera) {
    if !s.has do return
    HIGHLIGHT :: Color{1.0, 0.85, 0.10, 1.0}

    switch s.mode {
    case .Vertex:
        if int(s.index) >= len(m.vertices) do return
        p := m.vertices[s.index].position
        d := linalg.length(p - c.position) * 0.04
        // Three-axis cross marker.
        r.draw_line_overlay(p - {d,0,0}, p + {d,0,0}, HIGHLIGHT)
        r.draw_line_overlay(p - {0,d,0}, p + {0,d,0}, HIGHLIGHT)
        r.draw_line_overlay(p - {0,0,d}, p + {0,0,d}, HIGHLIGHT)
    case .Edge:
        if int(s.index) >= len(m.edges) do return
        e := m.edges[s.index]
        if e.halfEdge == NONE do return
        he := m.halfedges[e.halfEdge]
        a  := m.vertices[he.vert].position
        b  := m.vertices[m.halfedges[he.twin].vert].position
        r.draw_line_overlay(a, b, HIGHLIGHT)
    case .Face:
        if int(s.index) >= len(m.faces) do return
        f := m.faces[s.index]
        if f.halfEdge == NONE do return
        h := f.halfEdge
        for {
            next := m.halfedges[h].next
            a := m.vertices[m.halfedges[h].vert].position
            b := m.vertices[m.halfedges[next].vert].position
            r.draw_line_overlay(a, b, HIGHLIGHT)
            h = next
            if h == f.halfEdge do break
        }
    }
}

// Wireframe arrow from `from` to `to`. Five line segments: the shaft plus
// four arrowhead ticks, so the tip stays readable from any viewing angle
// (axis-aligned arrows degenerate to a single line at edge-on angles
// otherwise). `overlay` draws always-on-top via draw_line_overlay; pass
// false for in-scene arrows that should be occluded by geometry.
ARROW_HEAD_LEN_FRAC   :: f32(0.18) // head depth as a fraction of shaft length
ARROW_HEAD_WIDTH_FRAC :: f32(0.12) // head half-width as a fraction of shaft length

draw_arrow :: proc(r: ^Renderer, from, to: Vec3, color: Color, overlay: bool = true) {
    shaft  := to - from
    length := linalg.length(shaft)
    if length < 1e-6 do return
    dir := shaft / length

    // Pick any reference vector that isn't parallel to `dir` so the cross
    // product gives a usable perpendicular.
    ref := Vec3{0, 1, 0}
    if abs(linalg.dot(dir, ref)) > 0.95 do ref = {1, 0, 0}
    perp1 := linalg.normalize(linalg.cross(dir, ref))
    perp2 := linalg.cross(dir, perp1)

    head_len   := length * ARROW_HEAD_LEN_FRAC
    head_width := length * ARROW_HEAD_WIDTH_FRAC
    base := to - dir * head_len

    line :: proc(r: ^Renderer, a, b: Vec3, col: Color, overlay: bool) {
        if overlay do r.draw_line_overlay(a, b, col)
        else       do r.draw_line(a, b, col)
    }

    line(r, from, to, color, overlay)
    line(r, to, base + perp1 * head_width, color, overlay)
    line(r, to, base - perp1 * head_width, color, overlay)
    line(r, to, base + perp2 * head_width, color, overlay)
    line(r, to, base - perp2 * head_width, color, overlay)
}

// Draw the translation gizmo at `origin`. Each axis brightens when hovered
// or actively dragged.
draw_translation_gizmo :: proc(r: ^Renderer, ed: ^Editor, c: ^Camera, origin: Vec3) {
    size := gizmo_size(c, origin)
    base := [3]Color{
        {0.95, 0.30, 0.30, 1},
        {0.40, 0.95, 0.40, 1},
        {0.35, 0.55, 1.00, 1},
    }
    for a in 0..<3 {
        col := base[a]
        active := (ed.gizmo.dragging && ed.gizmo.axis == a) ||
                  (!ed.gizmo.dragging && ed.gizmo.hover_axis == a)
        if active do col = {1, 1, 0.2, 1}
        draw_arrow(r, origin, origin + axis_dir(a) * size, col)
    }
}
