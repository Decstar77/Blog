#pragma once
#include "sol_defines.h"
#include "sol_halfmesh.h"
#include "sol_list.h"
#include "sol_math.h"
#include "sol_render.h"

namespace sol {
    struct Primitive {
        HalfMesh        halfMesh;
        RenderMaterial  material;
        // Index into Renderer::staticMeshes. The RENDERER owns those buffers and
        // frees them at shutdown, which is why this is an index and not a
        // RenderStaticMesh by value - a copy here would be freed twice.
        // kNoPrimitive until the primitive has been uploaded.
        i32             renderMesh;
    };

    struct World {
        List<Primitive> primitives;
    };

    constexpr i32 kNoPrimitive = -1;

    bool RenderMeshFromHalfMesh( Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material, RenderStaticMesh * outMesh );
    i32  WorldAddPrimitive( World & world, Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material, const Mat4 & transform );
    bool WorldRebuildPrimitive( World & world, Renderer * r, i32 primitive );
    void WorldSetPrimitiveTransform( World & world, Renderer * r, i32 primitive, const Mat4 & transform );

    bool WorldCreateDefaultLevel( World & world, Renderer * r );
    void WorldFree( World & world );

} // namespace sol
