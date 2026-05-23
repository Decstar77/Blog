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
	normal:   Vec3,
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

Frame :: struct {
	normal: Vec3,
	t1:     Vec3,
	t2:     Vec3,
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

create_simplicial_set :: proc(allocator := context.temp_allocator) -> SimplicialSet {
	return SimplicialSet {
		verts = make(map[u32]struct{}, allocator),
		edges = make(map[u32]struct{}, allocator),
		faces = make(map[u32]struct{}, allocator),
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

outgoing_edges :: proc(m: ^HalfMesh, vi: u32) -> [dynamic]u32 {
	res := make([dynamic]u32, context.temp_allocator)

	hei := m.vertices[vi].halfEdge
	for {
		he := m.halfedges[hei]
		append(&res, he.edge)
		hei = m.halfedges[he.twin].next
		if hei == m.vertices[vi].halfEdge do break
	}

	return res
}

outgoing_edges_set :: proc(m: ^HalfMesh, vi: u32) -> SimplicialSet {
	res := create_simplicial_set()
	edges := outgoing_edges(m, vi)
	for edge in edges {
		set_add(&res.edges, edge)
	}
	return res
}

star_vertex :: proc(m: ^HalfMesh, vi: u32) -> SimplicialSet {
	res := create_simplicial_set()
	set_add(&res.verts, vi)

	vert := &m.vertices[vi]
	if vert.halfEdge == NONE do return res

	hei := vert.halfEdge
	for {
		he := m.halfedges[hei]

		set_add(&res.edges, he.edge)
		if he.face != NONE do set_add(&res.faces, he.face)

		assert(
			he.twin != NONE,
			"star_vertex: halfedge missing twin — builder_finish should have stitched a boundary twin for every interior halfedge",
		)
		next := m.halfedges[he.twin].next
		if next == NONE || next == vert.halfEdge do break
		hei = next
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

cache_face_normals :: proc(m: ^HalfMesh) {
	for &face in m.faces {
		v1i := m.halfedges[face.halfEdge].vert
		v2i := m.halfedges[m.halfedges[face.halfEdge].next].vert
		v3i := m.halfedges[m.halfedges[m.halfedges[face.halfEdge].next].next].vert

		v1 := m.vertices[v1i].position
		v2 := m.vertices[v2i].position
		v3 := m.vertices[v3i].position

		e1 := v2 - v1
		e2 := v3 - v1

		face.normal = linalg.normalize(linalg.cross(e1, e2))
	}
}

calucate_face_centroid :: proc(m: ^HalfMesh, fi: u32) -> Vec3 {
	hei := m.vertices[fi].halfEdge
	count := 0
	centriod := Vec3{0, 0, 0}
	for {
		he := m.halfedges[hei]
		centriod += m.vertices[he.vert].position
		count += 1
		hei = he.next
		if hei == m.vertices[fi].halfEdge do break
	}

	centriod /= f32(count)
	return centriod
}

calculate_face_barycentric_centers :: proc(m: ^HalfMesh) -> [dynamic]Vec3 {
	results := make([dynamic]Vec3, context.temp_allocator)
	for &face in m.faces {
		v1i := m.halfedges[face.halfEdge].vert
		v2i := m.halfedges[m.halfedges[face.halfEdge].next].vert
		v3i := m.halfedges[m.halfedges[m.halfedges[face.halfEdge].next].next].vert

		v1 := m.vertices[v1i].position
		v2 := m.vertices[v2i].position
		v3 := m.vertices[v3i].position

		center := (v1 + v2 + v3) / 3
		append(&results, center)
	}

	return results
}

angle_for_vertex_in_tri :: proc(m: ^HalfMesh, vi: u32, fi: u32) -> f32 {
	starti := NONE
	hei := m.faces[fi].halfEdge
	for {
		he := m.halfedges[hei]
		if he.vert == vi {
			starti = hei
			break
		}
		hei = he.next
		if hei == m.faces[fi].halfEdge do break
	}

	if starti == NONE {
		return 0
	}

	he0 := m.halfedges[starti]
	he1 := m.halfedges[he0.next]
	vi1 := he1.vert
	vi2 := m.halfedges[he1.next].vert
	e1 := m.vertices[vi1].position - m.vertices[vi].position
	e2 := m.vertices[vi2].position - m.vertices[vi].position
	angle := linalg.angle_between(e1, e2)
	return angle
}

area_for_tri :: proc(m: ^HalfMesh, fi: u32) -> f32 {
	he1 := m.halfedges[m.faces[fi].halfEdge]
	he2 := m.halfedges[he1.next]
	he3 := m.halfedges[he2.next]

	v1 := m.vertices[he1.vert].position
	v2 := m.vertices[he2.vert].position
	v3 := m.vertices[he3.vert].position

	e1 := v1 - v2
	e2 := v3 - v2

	area := 0.5 * linalg.length(linalg.cross(e2, e1))
	return area
}

calculate_vertex_normal_weighted_angle :: proc(m: ^HalfMesh) -> [dynamic]Vec3 {
	results := make([dynamic]Vec3, context.temp_allocator)
	for i in 0 ..< len(m.vertices) {
		starset := star_vertex(m, u32(i))

		total := f32(0)
		for facei in starset.faces {
			total += angle_for_vertex_in_tri(m, u32(i), facei)
		}

		if total <= 0 {
			append(&results, Vec3{0, 0, 0})
			continue
		}

		normal := Vec3{0, 0, 0}
		for facei in starset.faces {
			angle := angle_for_vertex_in_tri(m, u32(i), facei)
			normal += m.faces[facei].normal * (angle / total)
		}

		append(&results, linalg.normalize(normal))
	}
	return results
}

calculate_vertex_normal_weighted_face_area :: proc(m: ^HalfMesh, vi: u32) -> Vec3 {
	starset := star_vertex(m, vi)

	total := f32(0)
	for facei in starset.faces {
		total += area_for_tri(m, facei)
	}

	if total <= 0 {
		return Vec3{0, 0, 0}
	}

	normal := Vec3{0, 0, 0}
	for facei in starset.faces {
		area := area_for_tri(m, facei)
		normal += m.faces[facei].normal * (area / total)
	}

	r := linalg.normalize(normal)
	return r
}

calculate_vertex_normal_weighted_face_areas :: proc(m: ^HalfMesh) -> [dynamic]Vec3 {
	results := make([dynamic]Vec3, context.temp_allocator)
	for i in 0 ..< len(m.vertices) {
		normal := calculate_vertex_normal_weighted_face_area(m, u32(i))
		append(&results, normal)
	}
	return results
}

calculate_face_frame :: proc(m: ^HalfMesh, fi: u32) -> Frame {
	face := m.faces[fi]
	n := face.normal

	v1 := m.vertices[m.halfedges[face.halfEdge].vert].position
	v2 := m.vertices[m.halfedges[m.halfedges[face.halfEdge].next].vert].position
	t1 := linalg.normalize(v2 - v1)
	t2 := linalg.cross(t1, n)

	frame := Frame{}
	frame.normal = n
	frame.t1 = t1
	frame.t2 = t2
	return frame
}

calculate_face_frames :: proc(m: ^HalfMesh) -> [dynamic]Frame {
	frames := make([dynamic]Frame, context.temp_allocator)
	for face in m.faces {
		frame := calculate_face_frame(m, m.halfedges[face.halfEdge].face)
		append(&frames, frame)
	}
	return frames
}

calculate_vertex_frame :: proc(m: ^HalfMesh, vi: u32) -> Frame {
	n := calculate_vertex_normal_weighted_face_area(m, vi)

	v1 := m.vertices[vi].position
	v2 := m.vertices[m.halfedges[m.halfedges[m.vertices[vi].halfEdge].next].vert].position
	e := v2 - v1

	t1 := linalg.normalize(e - linalg.dot(n, e) * n)
	t2 := linalg.cross(t1, n)

	frame := Frame{}
	frame.normal = n
	frame.t1 = t1
	frame.t2 = t2
	return frame
}

calculate_vertex_frames :: proc(m: ^HalfMesh) -> [dynamic]Frame {
	frames := make([dynamic]Frame, context.temp_allocator)
	for vert in m.vertices {
		frame := calculate_vertex_frame(m, m.halfedges[vert.halfEdge].vert)
		append(&frames, frame)
	}
	return frames
}

// Returns the shared edge and the half-edge belonging to face1 along that
// edge. The half-edge is needed so callers can orient the edge consistently
// with face1's winding. Returns (NONE, NONE) if the faces share no edge.
find_common_edge :: proc(m: ^HalfMesh, f1i: u32, f2i: u32) -> (edge: u32, he1i: u32) {
	hei1 := m.faces[f1i].halfEdge
	for {
		he1 := m.halfedges[hei1]

		hei2 := m.faces[f2i].halfEdge
		for {
			he2 := m.halfedges[hei2]

			if he2.twin == hei1 {
				return he1.edge, hei1
			}

			hei2 = he2.next
			if hei2 == m.faces[f2i].halfEdge do break
		}

		hei1 = he1.next
		if hei1 == m.faces[f1i].halfEdge do break
	}

	return NONE, NONE
}

signed_dihedral_angle :: proc(m: ^HalfMesh, face1: u32, face2: u32) -> f32 {
	ei, hei1 := find_common_edge(m, face1, face2)
	if ei == NONE do return 0

	v1 := m.vertices[m.halfedges[hei1].vert].position
	v2 := m.vertices[m.halfedges[m.halfedges[hei1].next].vert].position
	d := v2 - v1
	if linalg.length(d) == 0 do return 0 // degenerate edge
	e := linalg.normalize(d)

	n1 := m.faces[face1].normal
	n2 := m.faces[face2].normal

	axis := linalg.cross(n1, e)
	x := linalg.dot(n1, n2)
	y := linalg.dot(axis, n2)

	angle := linalg.atan2(y, x)
	return angle
}

// Axis is assumed normalized
rotate_vector_about :: proc(v: Vec3, axis: Vec3, angle: f32) -> Vec3 {
	v_cos := v * linalg.cos(angle)
	v_cross := linalg.cross(axis, v) * linalg.sin(angle)
	v_dot := axis * linalg.dot(axis, v) * (1 - linalg.cos(angle))
	return v_cos + v_cross + v_dot
}

calculate_face_transport :: proc(m: ^HalfMesh, face1: u32, face2: u32, wt: Vec3) -> Vec3 {
	edge, hei1 := find_common_edge(m, face1, face2)
	if edge == NONE do return Vec3{0, 0, 0}

	v1 := m.vertices[m.halfedges[hei1].vert].position
	v2 := m.vertices[m.halfedges[m.halfedges[hei1].next].vert].position
	d := v2 - v1
	if linalg.length(d) == 0 do return Vec3{0, 0, 0} // degenerate edge
	axis := linalg.normalize(d)
	// signed_dihedral_angle returns the rotation about +axis that maps n2->n1,
	// so negate it to transport from face1's plane into face2's.
	angle := signed_dihedral_angle(m, face1, face2)
	ported := rotate_vector_about(wt, axis, -angle)
	return ported
}






