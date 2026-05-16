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
	ba = add_halfedge(m, HalfEdge{twin = ab, next = NONE, vert = b, edge = e, face = NONE})
	m.halfedges[ab].twin = ba
	m.edges[e].halfEdge = ab
	return
}

SimplicialSet :: struct {
	verts: map[u32]struct{},
	edges: map[u32]struct{},
	faces: map[u32]struct{},
}

create_simplicial_set :: proc() -> SimplicialSet {
	return SimplicialSet {
		verts = make(map[u32]struct{}),
		edges = make(map[u32]struct{}),
		faces = make(map[u32]struct{}),
	}
}

set_add :: proc( set : ^map[u32]struct{}, value : u32 ) {
	set[value] = {}
}

star_vertex :: proc(m: ^HalfMesh, vertexIndex: u32) -> SimplicialSet {
	vert := &m.vertices[vertexIndex]
	res := create_simplicial_set()
	hei := vert.halfEdge
	for {
		he := m.halfedges[hei]

		set_add(&res.verts, he.vert)
		set_add(&res.edges, he.edge)
		set_add(&res.faces, he.face)

		hei = m.halfedges[he.twin].next
		if (hei == vert.halfEdge) {break}
	}

	return res
}
