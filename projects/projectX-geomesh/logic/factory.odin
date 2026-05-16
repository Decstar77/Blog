package logic

import "core:math"
import "core:math/linalg"

halfmesh_to_triangles :: proc(m: ^HalfMesh, allocator := context.allocator) -> (positions: []Vec3, indices: []u32) {
	context.allocator = allocator

	positions = make([]Vec3, len(m.vertices))
	for v, i in m.vertices do positions[i] = v.position

	idx := make([dynamic]u32, 0, len(m.faces) * 3)
	for f in m.faces {
		start := f.halfEdge
		if start == NONE do continue
		anchor := m.halfedges[start].vert
		h := m.halfedges[start].next
		for {
			next := m.halfedges[h].next
			if next == start do break
			append(&idx, anchor, m.halfedges[h].vert, m.halfedges[next].vert)
			h = next
		}
	}
	indices = idx[:]
	return
}

// ---------------------------------------------------------------- builder --

// Polygon-soup -> half-edge mesh. Each call to `builder_add_face` registers
// one face by its CCW vertex ring; shared edges between faces are paired up
// via the (from,to) lookup table. After all faces are in, `builder_finish`
// stitches a boundary half-edge for every still-unpaired interior half-edge
// and chains them into boundary loops (face = NONE).
@(private="file")
Builder :: struct {
	m:        HalfMesh,
	edge_map: map[u64]u32, // (from << 32) | to  ->  inner half-edge index
}

@(private="file")
builder_init :: proc(b: ^Builder) {
	b.edge_map = make(map[u64]u32)
}

@(private="file")
ek :: proc(a, b: u32) -> u64 { return (u64(a) << 32) | u64(b) }

// Fan-triangulates any n-gon (n >= 3) from verts[0] so the produced mesh is
// always a pure triangle half-edge mesh.
@(private="file")
builder_add_face :: proc(b: ^Builder, verts: ..u32) {
	n := len(verts)
	if n < 3 do return
	for i in 1..<(n - 1) {
		builder_add_triangle(b, verts[0], verts[i], verts[i + 1])
	}
}

@(private="file")
builder_add_triangle :: proc(b: ^Builder, v0, v1, v2: u32) {
	verts := [3]u32{v0, v1, v2}
	n := 3
	f := add_face(&b.m)
	face_hes: [3]u32

	for i in 0..<n {
		a  := verts[i]
		bb := verts[(i + 1) % n]
		twin, has_twin := b.edge_map[ek(bb, a)]
		he: u32
		if has_twin {
			e := b.m.halfedges[twin].edge
			he = add_halfedge(&b.m, HalfEdge{twin = twin, next = NONE, vert = a, edge = e, face = f})
			b.m.halfedges[twin].twin = he
		} else {
			e := add_edge(&b.m)
			he = add_halfedge(&b.m, HalfEdge{twin = NONE, next = NONE, vert = a, edge = e, face = f})
			b.m.edges[e].halfEdge = he
		}
		b.edge_map[ek(a, bb)] = he
		face_hes[i] = he
		if b.m.vertices[a].halfEdge == NONE do b.m.vertices[a].halfEdge = he
	}
	for i in 0..<n {
		b.m.halfedges[face_hes[i]].next = face_hes[(i + 1) % n]
	}
	b.m.faces[f].halfEdge = face_hes[0]
}

@(private="file")
builder_finish :: proc(b: ^Builder) -> HalfMesh {
	// Add a boundary half-edge for every still-unpaired interior half-edge,
	// then thread the boundary loops. Assumes manifold input — each boundary
	// vertex has exactly one outgoing boundary half-edge.
	n_inner := len(b.m.halfedges)
	boundary_out := make(map[u32]u32) // vertex -> outgoing boundary half-edge
	defer delete(boundary_out)

	for i in 0..<n_inner {
		h := b.m.halfedges[i]
		if h.twin != NONE do continue
		from := h.vert
		to   := b.m.halfedges[h.next].vert
		bh := add_halfedge(&b.m, HalfEdge{twin = u32(i), next = NONE, vert = to, edge = h.edge, face = NONE})
		b.m.halfedges[i].twin = bh
		boundary_out[to] = bh
	}

	for i in n_inner..<len(b.m.halfedges) {
		bh := b.m.halfedges[i]
		// boundary half-edge from `bh.vert` ends at the origin of its inner twin
		end_vert := b.m.halfedges[bh.twin].vert
		if next_bh, ok := boundary_out[end_vert]; ok {
			b.m.halfedges[i].next = next_bh
		}
	}

	delete(b.edge_map)
	return b.m
}

// ----------------------------------------------------------------- plane --

create_plane :: proc(width: f32, height: f32) -> HalfMesh {
	hw := width  * 0.5
	hh := height * 0.5
	b: Builder
	builder_init(&b)

	v0 := add_vertex(&b.m, Vec3{-hw, 0, -hh})
	v1 := add_vertex(&b.m, Vec3{ hw, 0, -hh})
	v2 := add_vertex(&b.m, Vec3{ hw, 0,  hh})
	v3 := add_vertex(&b.m, Vec3{-hw, 0,  hh})

	builder_add_face(&b, v0, v1, v2, v3)
	return builder_finish(&b)
}

// ------------------------------------------------------------------ cube --

create_cube :: proc(size: f32) -> HalfMesh {
	s := size * 0.5
	b: Builder
	builder_init(&b)

	v := [8]u32{
		add_vertex(&b.m, Vec3{-s, -s, -s}), // 0
		add_vertex(&b.m, Vec3{ s, -s, -s}), // 1
		add_vertex(&b.m, Vec3{ s,  s, -s}), // 2
		add_vertex(&b.m, Vec3{-s,  s, -s}), // 3
		add_vertex(&b.m, Vec3{-s, -s,  s}), // 4
		add_vertex(&b.m, Vec3{ s, -s,  s}), // 5
		add_vertex(&b.m, Vec3{ s,  s,  s}), // 6
		add_vertex(&b.m, Vec3{-s,  s,  s}), // 7
	}

	// CCW winding viewed from outside.
	builder_add_face(&b, v[4], v[5], v[6], v[7]) // +Z front
	builder_add_face(&b, v[1], v[0], v[3], v[2]) // -Z back
	builder_add_face(&b, v[5], v[1], v[2], v[6]) // +X right
	builder_add_face(&b, v[0], v[4], v[7], v[3]) // -X left
	builder_add_face(&b, v[3], v[7], v[6], v[2]) // +Y top
	builder_add_face(&b, v[0], v[1], v[5], v[4]) // -Y bottom

	return builder_finish(&b)
}

// -------------------------------------------------------------- uv-sphere --

// `segments` = longitude divisions, `rings` = number of latitude bands
// (so `rings - 1` rings of vertices sit between the two poles).
create_uv_sphere :: proc(radius: f32, segments: int = 16, rings: int = 8) -> HalfMesh {
	seg := max(segments, 3)
	rng := max(rings, 2)
	b: Builder
	builder_init(&b)

	top := add_vertex(&b.m, Vec3{0, radius, 0})

	ring_start := make([]u32, rng - 1)
	defer delete(ring_start)
	for i in 0..<(rng - 1) {
		theta := f32(i + 1) / f32(rng) * math.PI
		sy := math.cos(theta)
		sr := math.sin(theta)
		first: u32
		for j in 0..<seg {
			phi := f32(j) / f32(seg) * 2 * math.PI
			p   := Vec3{sr * math.sin(phi), sy, sr * math.cos(phi)} * radius
			id  := add_vertex(&b.m, p)
			if j == 0 do first = id
		}
		ring_start[i] = first
	}

	bottom := add_vertex(&b.m, Vec3{0, -radius, 0})

	// Top cap (triangles).
	tr := ring_start[0]
	for j in 0..<seg {
		a  := tr + u32(j)
		bb := tr + u32((j + 1) % seg)
		builder_add_face(&b, top, a, bb)
	}

	// Middle quads. Wind (lower-left, lower-right, upper-right, upper-left)
	// so the outward normal points away from the y-axis.
	for i in 0..<(rng - 2) {
		r0 := ring_start[i]     // upper
		r1 := ring_start[i + 1] // lower
		for j in 0..<seg {
			j2 := u32((j + 1) % seg)
			ul := r0 + u32(j)
			ur := r0 + j2
			lr := r1 + j2
			ll := r1 + u32(j)
			builder_add_face(&b, ll, lr, ur, ul)
		}
	}

	// Bottom cap (triangles).
	br := ring_start[rng - 2]
	for j in 0..<seg {
		a  := br + u32(j)
		bb := br + u32((j + 1) % seg)
		builder_add_face(&b, bb, a, bottom)
	}

	return builder_finish(&b)
}

// -------------------------------------------------------------- ico-sphere --

@(private="file")
ico_midpoint :: proc(a, b: u32, verts: ^[dynamic]Vec3, cache: ^map[u64]u32) -> u32 {
	lo := min(a, b); hi := max(a, b)
	k  := (u64(lo) << 32) | u64(hi)
	if id, ok := cache[k]; ok do return id
	mp := linalg.normalize((verts[a] + verts[b]) * 0.5)
	id := u32(len(verts))
	append(verts, mp)
	cache[k] = id
	return id
}

create_ico_sphere :: proc(radius: f32, subdivisions: int = 0) -> HalfMesh {
	t := (1 + math.sqrt(f32(5))) * 0.5

	raw := [12]Vec3{
		{-1,  t,  0}, { 1,  t,  0}, {-1, -t,  0}, { 1, -t,  0},
		{ 0, -1,  t}, { 0,  1,  t}, { 0, -1, -t}, { 0,  1, -t},
		{ t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1},
	}
	base_tris := [20][3]u32{
		{ 0, 11,  5}, { 0,  5,  1}, { 0,  1,  7}, { 0,  7, 10}, { 0, 10, 11},
		{ 1,  5,  9}, { 5, 11,  4}, {11, 10,  2}, {10,  7,  6}, { 7,  1,  8},
		{ 3,  9,  4}, { 3,  4,  2}, { 3,  2,  6}, { 3,  6,  8}, { 3,  8,  9},
		{ 4,  9,  5}, { 2,  4, 11}, { 6,  2, 10}, { 8,  6,  7}, { 9,  8,  1},
	}

	verts := make([dynamic]Vec3)
	defer delete(verts)
	for v in raw do append(&verts, linalg.normalize(v))

	tris := make([dynamic][3]u32)
	defer delete(tris)
	for tr in base_tris do append(&tris, tr)

	for _ in 0..<subdivisions {
		cache := make(map[u64]u32)
		defer delete(cache)
		new_tris := make([dynamic][3]u32)
		for tri in tris {
			a, c, e := tri[0], tri[1], tri[2]
			ab := ico_midpoint(a, c, &verts, &cache)
			bc := ico_midpoint(c, e, &verts, &cache)
			ca := ico_midpoint(e, a, &verts, &cache)
			append(&new_tris, [3]u32{a, ab, ca})
			append(&new_tris, [3]u32{c, bc, ab})
			append(&new_tris, [3]u32{e, ca, bc})
			append(&new_tris, [3]u32{ab, bc, ca})
		}
		delete(tris)
		tris = new_tris
	}

	b: Builder
	builder_init(&b)
	for v in verts do add_vertex(&b.m, v * radius)
	for tri in tris do builder_add_face(&b, tri[0], tri[1], tri[2])
	return builder_finish(&b)
}

// ---------------------------------------------------------------- cylinder --

create_cylinder :: proc(radius: f32, height: f32, segments: int = 16) -> HalfMesh {
	seg := max(segments, 3)
	hh  := height * 0.5
	b: Builder
	builder_init(&b)

	top := make([]u32, seg); defer delete(top)
	bot := make([]u32, seg); defer delete(bot)
	for j in 0..<seg {
		phi := f32(j) / f32(seg) * 2 * math.PI
		x   := math.sin(phi) * radius
		z   := math.cos(phi) * radius
		bot[j] = add_vertex(&b.m, Vec3{x, -hh, z})
		top[j] = add_vertex(&b.m, Vec3{x,  hh, z})
	}

	// Side quads: (lower-left, lower-right, upper-right, upper-left).
	for j in 0..<seg {
		j2 := (j + 1) % seg
		builder_add_face(&b, bot[j], bot[j2], top[j2], top[j])
	}

	// Top n-gon, wound CCW as seen from +Y (so outward = +Y).
	top_face := make([]u32, seg); defer delete(top_face)
	for j in 0..<seg do top_face[j] = top[seg - 1 - j]
	builder_add_face(&b, ..top_face)

	// Bottom n-gon, wound CCW as seen from -Y (so outward = -Y).
	bot_face := make([]u32, seg); defer delete(bot_face)
	for j in 0..<seg do bot_face[j] = bot[j]
	builder_add_face(&b, ..bot_face)

	return builder_finish(&b)
}
