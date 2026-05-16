package logic

import "core:fmt"
import "core:math"
import "core:math/linalg"

Mesh_Handle :: distinct u32

INVALID_MESH :: Mesh_Handle(0)

Mat4 :: matrix[4, 4]f32

Renderer :: struct {
	create_mesh:         proc(positions: []Vec3, indices: []u32) -> Mesh_Handle,
	update_mesh:         proc(mesh: Mesh_Handle, positions: []Vec3),
	draw_mesh:           proc(mesh: Mesh_Handle, color: Color),
	draw_mesh_xform:     proc(mesh: Mesh_Handle, model: Mat4, color: Color),
	draw_triangles:      proc(positions: []Vec3, indices: []u32, color: Color),
	draw_line:           proc(a, b: Vec3, color: Color),
	draw_line_overlay:   proc(a, b: Vec3, color: Color),
	clear:               proc(color: Color),
	set_viewport:        proc(w, h: int),
	set_view_projection: proc(vp: Mat4),
	load_font:           proc(png_path, json_path: string) -> Font_Handle,
	draw_text_3d:        proc(
		font: Font_Handle,
		text: string,
		anchor: Vec3,
		em_size: f32,
		color: Color,
		cam_right, cam_up: Vec3,
	),
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
	aspect:                     f32, // viewport w/h, used to refresh proj

	// Editor input. mouse_x/y are absolute cursor pixels (top-left origin),
	// updated whether or not look-mode is active. lmb is held-state; the
	// editor does its own edge detection.
	mouse_x, mouse_y:           f32,
	viewport_w, viewport_h:     f32,
	lmb:                        bool,
	k1, k2, k3:                 bool, // 1/2/3 select Vertex/Edge/Face mode
}

Camera :: struct {
	position:   Vec3,
	yaw, pitch: f32,
	fov_y:      f32,
	aspect:     f32,
	near, far:  f32,
	move_speed: f32,
	look_speed: f32,
}

App :: struct {
	renderer:      Renderer,
	font:          Font_Handle,
	camera:        Camera,
	input:         Input,
	editor:        Editor,
	time:          f32,
	halfmesh:      HalfMesh,
	simplicial:    SimplicialSet,
	mesh_handle:   Mesh_Handle,
	sphere_mesh:   Mesh_Handle,
	cylinder_mesh: Mesh_Handle,
}

initialize :: proc(app: ^App) {
	app.halfmesh = create_cube(3) //create_plane(4, 4)

	positions, indices := halfmesh_to_triangles(&app.halfmesh)
	defer delete(positions)
	defer delete(indices)
	app.mesh_handle = app.renderer.create_mesh(positions, indices)

	// Template unit primitives reused for every simplicial vertex / edge draw.
	app.sphere_mesh = upload_template_mesh(&app.renderer, create_uv_sphere(1.0, 12, 8))
	app.cylinder_mesh = upload_template_mesh(&app.renderer, create_cylinder(1.0, 1.0, 12))

	app.font = app.renderer.load_font(
		"res/fonts/FiraCode-Regular.png",
		"res/fonts/FiraCode-Regular.json",
	)

	app.camera = Camera {
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
	cy := math.cos(c.yaw); sy := math.sin(c.yaw)
	return {sy * cp, sp, -cy * cp}
}

camera_right :: proc(c: ^Camera) -> Vec3 {
	cy := math.cos(c.yaw); sy := math.sin(c.yaw)
	return {cy, 0, sy}
}

camera_view_projection :: proc(c: ^Camera) -> Mat4 {
	fwd := camera_forward(c)
	eye := c.position
	view := linalg.matrix4_look_at_f32(eye, eye + fwd, {0, 1, 0})
	proj := linalg.matrix4_perspective_f32(c.fov_y, c.aspect, c.near, c.far)
	return proj * view
}

update_camera :: proc(c: ^Camera, inp: Input, dt: f32) {
	if inp.aspect > 0 do c.aspect = inp.aspect

	if inp.scroll_dy != 0 {
		c.move_speed = clamp(c.move_speed * math.pow(f32(1.2), inp.scroll_dy), 0.1, 500)
	}

	if !inp.look_active do return

	c.yaw += inp.mouse_dx * c.look_speed
	c.pitch -= inp.mouse_dy * c.look_speed
	limit := f32(math.PI * 0.49)
	c.pitch = clamp(c.pitch, -limit, limit)

	fwd := camera_forward(c)
	right := camera_right(c)
	move := Vec3{}
	if inp.forward do move += fwd
	if inp.back do move -= fwd
	if inp.right do move += right
	if inp.left do move -= right
	if inp.up do move += {0, 1, 0}
	if inp.down do move += {0, -1, 0}

	if move != {0, 0, 0} {
		move = linalg.normalize(move)
		speed := c.move_speed * (inp.boost ? 4.0 : 1.0)
		c.position += move * speed * dt
	}
}

halfedge_draw_halfedges :: proc(app: ^App, hm: ^HalfMesh) {
	for h in hm.halfedges {
		p1 := hm.vertices[h.vert].position
		p2 := hm.vertices[hm.halfedges[h.next].vert].position
		p3 := p2 + (p1 - p2) * 0.9
		p4 := p1 + (p2 - p1) * 0.9
		p3.y += 0.1
		p4.y += 0.1
		draw_arrow(&app.renderer, p3, p4, {1, 1, 1, 1}, false)
	}
}

upload_template_mesh :: proc(r: ^Renderer, hm: HalfMesh) -> Mesh_Handle {
	m := hm
	defer {
		delete(m.vertices)
		delete(m.edges)
		delete(m.faces)
		delete(m.halfedges)
	}
	pos, idx := halfmesh_to_triangles(&m)
	defer delete(pos)
	defer delete(idx)
	return r.create_mesh(pos, idx)
}

// Build a row-major Mat4 literal for `T(t) * R(basis_x, basis_y, basis_z) * S(s)`.
@(private = "file")
trs_matrix :: proc(t: Vec3, bx, by, bz: Vec3, s: Vec3) -> Mat4 {
	return Mat4 {
		s.x * bx.x,
		s.y * by.x,
		s.z * bz.x,
		t.x,
		s.x * bx.y,
		s.y * by.y,
		s.z * bz.y,
		t.y,
		s.x * bx.z,
		s.y * by.z,
		s.z * bz.z,
		t.z,
		0,
		0,
		0,
		1,
	}
}

// Rotation basis that maps the unit cylinder's +Y axis onto `axis` (must be unit).
// Returns three orthonormal columns (new_x, new_y=axis, new_z).
@(private = "file")
basis_from_y :: proc(axis: Vec3) -> (bx, by, bz: Vec3) {
	by = axis
	ref := Vec3{1, 0, 0}
	if abs(linalg.dot(by, ref)) > 0.95 do ref = {0, 0, 1}
	bx = linalg.normalize(linalg.cross(by, ref))
	bz = linalg.cross(bx, by)
	return
}

draw_simplicial_set :: proc(
	r: ^Renderer,
	m: ^HalfMesh,
	set: ^SimplicialSet,
	c: ^Camera,
	sphere, cylinder: Mesh_Handle,
) {
	GREY :: Color{0.9, 0.9, 0.9, 1.0}
	YELLOW :: Color{0.85, 0.85, 0.20, 1.0}
	FACE_FILL :: Color{0.85, 0.75, 0.10, 0.85}
	V_RADIUS :: f32(0.08)
	E_RADIUS :: f32(0.03)

	for v_idx in set.verts {
		if v_idx == NONE || int(v_idx) >= len(m.vertices) do continue
		p := m.vertices[v_idx].position
		model := trs_matrix(p, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {V_RADIUS, V_RADIUS, V_RADIUS})
		r.draw_mesh_xform(sphere, model, GREY)
	}

	for e_idx in set.edges {
		if e_idx == NONE || int(e_idx) >= len(m.edges) do continue
		e := m.edges[e_idx]
		if e.halfEdge == NONE do continue
		he := m.halfedges[e.halfEdge]
		a := m.vertices[he.vert].position
		b := m.vertices[m.halfedges[he.twin].vert].position
		d := b - a
		L := linalg.length(d)
		if L < 1e-6 do continue
		bx, by, bz := basis_from_y(d / L)
		mid := (a + b) * 0.5
		model := trs_matrix(mid, bx, by, bz, {E_RADIUS, L, E_RADIUS})
		r.draw_mesh_xform(cylinder, model, YELLOW)
	}

	// Faces: triangulate each highlighted face (fan from anchor), then nudge
	// vertices along the face normal so the patch sits just above the base
	// mesh. Emitted as triangle soup since each face has its own normal —
	// shared edge vertices need separate offset copies per face.
	NORMAL_OFFSET :: f32(0.002)
	positions := make([dynamic]Vec3, 0, len(set.faces) * 6)
	indices := make([dynamic]u32, 0, len(set.faces) * 6)
	defer delete(positions)
	defer delete(indices)

	for f_idx in set.faces {
		if f_idx == NONE || int(f_idx) >= len(m.faces) do continue
		f := m.faces[f_idx]
		start := f.halfEdge
		if start == NONE do continue
		anchor_v := m.halfedges[start].vert
		pa := m.vertices[anchor_v].position
		// Face normal from the first triangle of the fan.
		h0 := m.halfedges[start].next
		h1 := m.halfedges[h0].next
		p0 := m.vertices[m.halfedges[h0].vert].position
		p1 := m.vertices[m.halfedges[h1].vert].position
		n := linalg.cross(p0 - pa, p1 - pa)
		nl := linalg.length(n)
		if nl < 1e-6 do continue
		offset := n * (NORMAL_OFFSET / nl)

		h := h0
		for {
			next := m.halfedges[h].next
			if next == start do break
			pb := m.vertices[m.halfedges[h].vert].position
			pc := m.vertices[m.halfedges[next].vert].position
			base := u32(len(positions))
			append(&positions, pa + offset, pb + offset, pc + offset)
			append(&indices, base, base + 1, base + 2)
			h = next
		}
	}
	r.draw_triangles(positions[:], indices[:], FACE_FILL)
}

draw_normals :: proc(app: ^App) {
	cache_face_normals(&app.halfmesh)
	centers := calculate_face_barycentric_centers(&app.halfmesh)
	for i in 0 ..< len(centers) {
		col := Color{0.2, 0.2, 1.0, 1.0}
		app.renderer.draw_line(centers[i], centers[i] + app.halfmesh.faces[i].normal * 0.1, col)
	}

	verts: = calculate_vertex_normal_weighted_face_area(&app.halfmesh)
	for i in 0 ..< len(verts) {
		col := Color{0.2, 0.2, 1.0, 1.0}
		v := app.halfmesh.vertices[i].position
		app.renderer.draw_line(v, v + verts[i] * 0.5, col)
	}
}

frame :: proc(app: ^App, dt: f32) {
	app.time += dt

	update_camera(&app.camera, app.input, dt)

	if editor_update(&app.editor, &app.halfmesh, &app.camera, app.input) {
		positions, indices := halfmesh_to_triangles(&app.halfmesh)
		defer delete(positions)
		defer delete(indices)
		if app.renderer.update_mesh != nil {
			app.renderer.update_mesh(app.mesh_handle, positions)
		}
	}

	app.renderer.clear({0.10, 0.11, 0.15, 1.0})
	app.renderer.set_view_projection(camera_view_projection(&app.camera))

	app.renderer.draw_mesh(app.mesh_handle, {0.45, 0.55, 0.85, 1.0})
	draw_normals(app)

	app.simplicial = star_vertex(&app.halfmesh, 7)
	//app.simplicial = star_edge(&app.halfmesh, 3)
	//app.simplicial = star_face(&app.halfmesh, 0)
	//app.simplicial = closure_vertex(&app.halfmesh, 7)
	//app.simplicial = closure_edge(&app.halfmesh, 3)
	//app.simplicial = closure_face(&app.halfmesh, 0)
	//app.simplicial = link_vertex(&app.halfmesh, 7)
	//app.simplicial = link_edge(&app.halfmesh, 3)
	//app.simplicial = link_face(&app.halfmesh, 0)
	//app.simplicial = outgoing_edges_set(&app.halfmesh, 7)

	draw_simplicial_set(
		&app.renderer,
		&app.halfmesh,
		&app.simplicial,
		&app.camera,
		app.sphere_mesh,
		app.cylinder_mesh,
	)

	//halfedge_draw_halfedges(app, &app.halfmesh)

	edge_color := Color{1, 1, 1, 1}
	for e in app.halfmesh.edges {
		if e.halfEdge == NONE do continue
		he := app.halfmesh.halfedges[e.halfEdge]
		a := app.halfmesh.vertices[he.vert].position
		b := app.halfmesh.vertices[app.halfmesh.halfedges[he.twin].vert].position
		app.renderer.draw_line(a, b, edge_color)
	}

	app.renderer.draw_line({0, 0, 0}, {1, 0, 0}, {0.95, 0.30, 0.30, 1})
	app.renderer.draw_line({0, 0, 0}, {0, 1, 0}, {0.30, 0.95, 0.30, 1})
	app.renderer.draw_line({0, 0, 0}, {0, 0, 1}, {0.30, 0.55, 0.95, 1})

	cam_fwd := camera_forward(&app.camera)
	cam_right := linalg.normalize(linalg.cross(cam_fwd, Vec3{0, 1, 0}))
	cam_up := linalg.normalize(linalg.cross(cam_right, cam_fwd))

	draw_selection_highlight(&app.renderer, &app.halfmesh, app.editor.selection, &app.camera)
	if origin, ok := selection_centroid(&app.halfmesh, app.editor.selection); ok {
		draw_translation_gizmo(&app.renderer, &app.editor, &app.camera, origin)
	}

	label_color := Color{1, 1, 1, 1}
	for v, i in app.halfmesh.vertices {
		buf: [16]u8
		text := fmt.bprintf(buf[:], "v{}", i)
		anchor := v.position + {0, 0.08, 0}
		app.renderer.draw_text_3d(app.font, text, anchor, 0.2, label_color, cam_right, cam_up)
	}

	free_all(context.temp_allocator)
}
