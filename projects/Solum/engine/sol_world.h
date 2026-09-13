#pragma once
#include "sol_math.h"
#include "sol_list.h"
#include "sol_render.h"
#include "sol_halfmesh.h"

namespace sol {
    struct Primitive {
        HalfMesh            halfMesh;
        RenderStaticMesh    mesh;
        RenderMaterial      material;
    };

    struct World {
        List<Primitive> primitives;
    };
}


