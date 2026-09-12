#pragma once
#include "sol_math.h"
#include "sol_list.h"
#include "sol_render.h"

namespace sol {
    struct HMVertex {
        Vec3    position;
        i32     halfEdge;
    };

    struct HMEdge {
        i32     halfEdge;
    };

    struct HMFace {
        i32     halfEdge;
        Vec3    normal;
    };

    struct HalfEdge {
        i32 next;
        i32 twin;
        i32 vert;// vertex this half-edge points FROM (origin)
        i32 edge;
        i32 face;// NONE for boundary half-edges
    };

    struct HalfMesh {
        List<HMVertex>  vertices;
        List<HMEdge>    edges;
        List<HMFace>    faces;
        List<HalfEdge>  halfEdges;
    };

    void HalfMeshCreateTriangle( HalfMesh & mesh );
    void HalfMeshCreateCube( HalfMesh & mesh, i32 size );

    struct Primitive {
        HalfMesh            halfMesh;
        RenderStaticMesh    mesh;
        RenderMaterial      material;
    };

    struct World {
        List<Primitive> primitives;
    };
}


