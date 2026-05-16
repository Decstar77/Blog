package logic

import "core:crypto/legacy/sha1"
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

set_add :: proc(set: ^map[u32]struct{}, value: u32) {set[value] = {}}
set_union :: proc(dst: ^map[u32]struct{}, src: map[u32]struct{}) {for k in src do dst[k] = {}}
set_difference :: proc(dst: ^map[u32]struct{}, src: map[u32]struct{}) {for k in src do delete_key(dst, k)}
set_intersection :: proc(dst: ^map[u32]struct{}, other: map[u32]struct{}) {
	for k in dst {
		if k not_in other do delete_key(dst, k)
	}
}

create_simplicial_set :: proc() -> SimplicialSet {
	return SimplicialSet {
		verts = make(map[u32]struct{}),
		edges = make(map[u32]struct{}),
		faces = make(map[u32]struct{}),
	}
}

simplicial_set_union :: proc(dst: ^SimplicialSet, src: SimplicialSet) -> ^SimplicialSet {
	set_union(&dst.verts, src.verts)
	set_union(&dst.edges, src.edges)
	set_union(&dst.faces, src.faces)
	return dst
}

simplicial_set_difference :: proc(dst: ^SimplicialSet, src: SimplicialSet) -> ^SimplicialSet {
	set_difference(&dst.verts, src.verts)
	set_difference(&dst.edges, src.edges)
	set_difference(&dst.faces, src.faces)
	return dst
}

simplicial_set_intersection :: proc(dst: ^SimplicialSet, src: SimplicialSet) -> ^SimplicialSet {
	set_intersection(&dst.verts, src.verts)
	set_intersection(&dst.edges, src.edges)
	set_intersection(&dst.faces, src.faces)
	return dst
}

star_vertex :: proc(m: ^HalfMesh, vi: u32) -> SimplicialSet {
	vert := &m.vertices[vi]
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

star_edge :: proc(m: ^HalfMesh, ei: u32) -> SimplicialSet {
	res := create_simplicial_set()
	set_add(&res.edges, ei)
	he1 := m.halfedges[m.edges[ei].halfEdge]
	he2 := m.halfedges[he1.twin]
	if (he1.face != NONE) do set_add(&res.faces, he1.face)
	if (he2.face != NONE) do set_add(&res.faces, he2.face)
	return res
}

star_face :: proc(m: ^HalfMesh, fi: u32) -> SimplicialSet {
	res := create_simplicial_set()
	set_add(&res.faces, fi)
	return res
}

star :: proc(m: ^HalfMesh, s: SimplicialSet) -> SimplicialSet {
	res := create_simplicial_set()
	for v in s.verts do simplicial_set_union(&res, star_vertex(m, v))
	for e in s.edges do simplicial_set_union(&res, star_edge(m, e))
	for f in s.faces do simplicial_set_union(&res, star_face(m, f))
	return res
}

closure_vertex :: proc(m: ^HalfMesh, vi: u32) -> SimplicialSet {
	res := create_simplicial_set()
	set_add(&res.verts, vi)
	return res
}

closure_edge :: proc(m: ^HalfMesh, ei: u32) -> SimplicialSet {
	res := create_simplicial_set()
	set_add(&res.edges, ei)
	he1 := m.halfedges[m.edges[ei].halfEdge]
	he2 := m.halfedges[he1.twin]
	set_add(&res.verts, he1.vert)
	set_add(&res.verts, he2.vert)
	return res
}

closure_face :: proc(m: ^HalfMesh, fi: u32) -> SimplicialSet {
	res := create_simplicial_set()
	set_add(&res.faces, fi)
	hei := m.faces[fi].halfEdge
	for {
		he := m.halfedges[hei]
		set_add(&res.verts, he.vert)
		set_add(&res.edges, he.edge)
		hei = he.next
		if (hei == m.faces[fi].halfEdge) do break
	}

	return res
}

closure :: proc(m: ^HalfMesh, s: SimplicialSet) -> SimplicialSet {
	res := create_simplicial_set()
	for v in s.verts do simplicial_set_union(&res, closure_vertex(m, v))
	for e in s.edges do simplicial_set_union(&res, closure_edge(m, e))
	for f in s.faces do simplicial_set_union(&res, closure_face(m, f))
	return res
}

link :: proc(m: ^HalfMesh, s: SimplicialSet) -> SimplicialSet {
	res := closure(m, star(m, s))
	simplicial_set_difference(&res, star(m, closure(m, s)))
	return res
}

link_vertex :: proc(m: ^HalfMesh, vi: u32) -> SimplicialSet {
	s := create_simplicial_set()
	set_add(&s.verts, vi)
	return link(m, s)
}

link_edge :: proc(m: ^HalfMesh, ei: u32) -> SimplicialSet {
	s := create_simplicial_set()
	set_add(&s.edges, ei)
	return link(m, s)
}

link_face :: proc(m: ^HalfMesh, fi: u32) -> SimplicialSet {
	s := create_simplicial_set()
	set_add(&s.faces, fi)
	return link(m, s)
}
