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
        // The RENDERER owns the mesh's buffers and frees them at shutdown,
        // which is why this is a handle and not a RenderStaticMesh by value -
        // a copy here would be freed twice. Null until the primitive has been
        // uploaded, and null again if the mesh it named was destroyed.
        RenderMeshHandle    renderMesh;
    };

    struct World {
        List<Primitive> primitives;
        // kNoPrimitive when nothing is selected. Note that a zero-initialised
        // World would leave this pointing at primitive 0, which is why
        // WorldCreate exists - use it rather than {}.
        i32             selected;
    };

    constexpr i32 kNoPrimitive = -1;

    // Warm and bright, multiplied into the fragment colour, so a selected
    // object still shows the shading that says which way its faces point.
    constexpr Vec4 kSelectionTint = { 1.9f, 1.35f, 0.55f, 1.0f };

    // An empty world with nothing selected.
    World WorldCreate();

    // Triangulates the half-mesh and uploads it. Null handle on failure.
    RenderMeshHandle RenderMeshFromHalfMesh( Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material );
    i32  WorldAddPrimitive( World & world, Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material, const Mat4 & transform );
    bool WorldRebuildPrimitive( World & world, Renderer * r, i32 primitive );
    void WorldSetPrimitiveTransform( World & world, Renderer * r, i32 primitive, const Mat4 & transform );

    // Nearest primitive the ray meets, tested against world-space bounding
    // boxes rather than triangles - close enough for boxes and planes, and it
    // costs one pass over the vertices instead of one per face.
    bool WorldPick( const World & world, Renderer * r, Vec3 rayOrigin, Vec3 rayDirection,
                    i32 * outPrimitive );

    // Moves the highlight. kNoPrimitive clears it. Only touches the tint, so
    // there is no GPU work and nothing to synchronise.
    void WorldSetSelected( World & world, Renderer * r, i32 primitive );

    // Frees the primitive's render mesh and its half-mesh, then closes the gap
    // it leaves. False for an index that names nothing.
    //
    // Removal is order preserving, so every primitive after this one shifts
    // down by one and any index held across the call stops naming what it did.
    // The selection is fixed up here; anything else holding an index has to run
    // it through WorldRemapPrimitive.
    bool WorldRemovePrimitive( World & world, Renderer * r, i32 primitive );

    // Where an index held across a WorldRemovePrimitive lands afterwards, or
    // kNoPrimitive if it named the primitive that was removed.
    i32  WorldRemapPrimitive( i32 held, i32 removed );

    bool WorldCreateDefaultLevel( World & world, Renderer * r );
    void WorldFree( World & world );

} // namespace sol
