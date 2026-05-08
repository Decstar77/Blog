package logic

import "core:math"
import "core:math/linalg"

Vec3 :: [3]f32
Color :: [4]f32

NONE :: max(u32)

Vertex :: struct {
	position: Vec3,
	halfEdge: u32,
}

Edge :: struct {
	halfEdge: u32,
}

Face :: struct {
	halfEdge: u32,
}

HalfEdge :: struct {
	twin: u32,
	next: u32,
	vert: u32, // vertex this half-edge points FROM (origin)
	edge: u32,
	face: u32, // NONE for boundary half-edges
}

HalfMesh :: struct {
	vertices:  [dynamic]Vertex,
	edges:     [dynamic]Edge,
	faces:     [dynamic]Face,
	halfedges: [dynamic]HalfEdge,
}

add_vertex :: proc(m: ^HalfMesh, p: Vec3) -> u32 {
	id := u32(len(m.vertices))
	append(&m.vertices, Vertex{position = p, halfEdge = NONE})
	return id
}

add_edge :: proc(m: ^HalfMesh) -> u32 {
	id := u32(len(m.edges))
	append(&m.edges, Edge{halfEdge = NONE})
	return id
}

add_face :: proc(m: ^HalfMesh) -> u32 {
	id := u32(len(m.faces))
	append(&m.faces, Face{halfEdge = NONE})
	return id
}

add_halfedge :: proc(m: ^HalfMesh, he: HalfEdge) -> u32 {
	id := u32(len(m.halfedges))
	append(&m.halfedges, he)
	return id
}

add_edge_pair :: proc(m: ^HalfMesh, a, b: u32) -> (ab: u32, ba: u32) {
	e := add_edge(m)
	ab = add_halfedge(m, HalfEdge{twin = NONE, next = NONE, vert = a, edge = e, face = NONE})
	ba = add_halfedge(m, HalfEdge{twin = ab,   next = NONE, vert = b, edge = e, face = NONE})
	m.halfedges[ab].twin = ba
	m.edges[e].halfEdge = ab
	return
}

// ---- construction ----------------------------------------------------------

// Build a flat XZ-plane quad of `width` x `height` centered on the origin.
//
//   v3 ---- v2
//    |  f0  |    (one quad face, CCW when viewed from +Y)
//   v0 ---- v1
//
// Half-edges around f0: h01 -> h12 -> h23 -> h30 -> h01
// Boundary loop (face = NONE): h10 -> h03 -> h32 -> h21 -> h10
create_plane :: proc(width: f32, height: f32) -> HalfMesh {
	m := HalfMesh{}

	hw := width  * 0.5
	hh := height * 0.5

	v0 := add_vertex(&m, Vec3{-hw, 0, -hh})
	v1 := add_vertex(&m, Vec3{ hw, 0, -hh})
	v2 := add_vertex(&m, Vec3{ hw, 0,  hh})
	v3 := add_vertex(&m, Vec3{-hw, 0,  hh})

	f0 := add_face(&m)

	// Four edges, each with its twin pair.
	h01, h10 := add_edge_pair(&m, v0, v1)
	h12, h21 := add_edge_pair(&m, v1, v2)
	h23, h32 := add_edge_pair(&m, v2, v3)
	h30, h03 := add_edge_pair(&m, v3, v0)

	// Inner loop: assign face and `next` pointers.
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

	// Outgoing half-edge for each vertex.
	m.vertices[v0].halfEdge = h01
	m.vertices[v1].halfEdge = h12
	m.vertices[v2].halfEdge = h23
	m.vertices[v3].halfEdge = h30

	return m
}

// ---- conversion ------------------------------------------------------------

// Triangulate every face of `m` with a fan from its first vertex.
// Caller owns both returned slices.
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

// Iterate unique edges, calling `visit` once per edge with its two endpoint
// positions. Cheap helper for wireframe overlays.
halfmesh_for_each_edge :: proc(m: ^HalfMesh, visit: proc(a, b: Vec3, user: rawptr), user: rawptr) {
	for e in m.edges {
		if e.halfEdge == NONE do continue
		he := m.halfedges[e.halfEdge]
		a  := m.vertices[he.vert].position
		b  := m.vertices[m.halfedges[he.twin].vert].position
		visit(a, b, user)
	}
}