#pragma once
#include "sol_defines.h"
#include "sol_halfmesh.h"
#include "sol_list.h"
#include "sol_math.h"
#include "sol_render.h"

namespace sol {
    struct Primitive {
        HalfMesh            halfMesh;
        RenderMaterial      material;
        RenderMeshHandle    renderMesh;
    };

    constexpr i32 kNoPrimitive = -1;
    constexpr Vec4 kSelectionTint = { 1.9f, 1.35f, 0.55f, 1.0f };

    struct World {
        List<Primitive> primitives;
        i32             selected; // kNoPrimitive when nothing is selected.
    };

    World WorldCreate();

    RenderMeshHandle    RenderMeshFromHalfMesh( Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material );
    i32                 WorldAddPrimitive( World & world, Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material, const Mat4 & transform );
    bool                WorldRebuildPrimitive( World & world, Renderer * r, i32 primitive );
    void                WorldSetPrimitiveTransform( World & world, Renderer * r, i32 primitive, const Mat4 & transform );

    bool                WorldPick( const World & world, Renderer * r, Vec3 rayOrigin, Vec3 rayDirection, i32 * outPrimitive );
    void                WorldSetSelected( World & world, Renderer * r, i32 primitive );
    void                WorldSetPrimitiveHighlight( World & world, Renderer * r, i32 primitive, bool highlight );
    bool                WorldSetEditOverlay( World & world, Renderer * r, i32 primitive );
    bool                WorldRemovePrimitive( World & world, Renderer * r, i32 primitive );
    i32                 WorldRemapPrimitive( i32 held, i32 removed );

    bool                WorldCreateDefaultLevel( World & world, Renderer * r );
    void                WorldFree( World & world );

} // namespace sol
