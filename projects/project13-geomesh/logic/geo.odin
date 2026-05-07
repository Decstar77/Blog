package logic

import "core:math"
import "core:math/linalg"

Vec3 :: [3]f32
Color :: [4]f32

Edge :: struct {
    i : i32,
    j : i32
}

Face :: struct {
    i : i32,
    j : i32,
    k : i32
}

Mesh :: struct {
	vertices: [dynamic]Vec3,
	edges:    [dynamic]Edge,
	faces:    [dynamic]Face,
}

create_plane :: proc(width: f32, height: f32) -> Mesh {
	mesh := Mesh{}

	mesh.vertices = make([dynamic]Vec3, 4)
	mesh.edges = make([dynamic]Edge, 5)
	mesh.faces = make([dynamic]Face, 2)

    mesh.vertices[0] = Vec3{ -1, -1, 0 }
    mesh.vertices[1] = Vec3{ -1,  1, 0 }
    mesh.vertices[2] = Vec3{  1,  1, 0 }
    mesh.vertices[3] = Vec3{  1, -1, 0 }

    mesh.edges[0] = Edge{ i=0, j=1 }
    mesh.edges[1] = Edge{ i=1, j=2 }
    mesh.edges[2] = Edge{ i=2, j=3 }
    mesh.edges[3] = Edge{ i=3, j=0 }
    mesh.edges[4] = Edge{ i=3, j=1 }

    mesh.faces[0] = Face{ i=0, j=1, k=4 }
    mesh.faces[1] = Face{ i=0, j=2, k=3 }

	return mesh
}
