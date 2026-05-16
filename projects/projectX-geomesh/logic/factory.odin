package logic

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

create_plane :: proc(width: f32, height: f32) -> HalfMesh {
	m := HalfMesh{}

	hw := width  * 0.5
	hh := height * 0.5

	v0 := add_vertex(&m, Vec3{-hw, 0, -hh})
	v1 := add_vertex(&m, Vec3{ hw, 0, -hh})
	v2 := add_vertex(&m, Vec3{ hw, 0,  hh})
	v3 := add_vertex(&m, Vec3{-hw, 0,  hh})

	f0 := add_face(&m)

	h01, h10 := add_edge_pair(&m, v0, v1)
	h12, h21 := add_edge_pair(&m, v1, v2)
	h23, h32 := add_edge_pair(&m, v2, v3)
	h30, h03 := add_edge_pair(&m, v3, v0)

	inner := [4]u32{h01, h12, h23, h30}
	for he in inner {
		m.halfedges[he].face = f0
	}
	m.halfedges[h01].next = h12
	m.halfedges[h12].next = h23
	m.halfedges[h23].next = h30
	m.halfedges[h30].next = h01
	m.faces[f0].halfEdge = h01

	// Outer (boundary) loop: face stays NONE, walk the perimeter the other way.
	// h10 (v1->v0) -> h03 (v0->v3) -> h32 (v3->v2) -> h21 (v2->v1) -> h10
	m.halfedges[h10].next = h03
	m.halfedges[h03].next = h32
	m.halfedges[h32].next = h21
	m.halfedges[h21].next = h10

	m.vertices[v0].halfEdge = h01
	m.vertices[v1].halfEdge = h12
	m.vertices[v2].halfEdge = h23
	m.vertices[v3].halfEdge = h30

	return m
}


