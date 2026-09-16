#pragma once
#include "sol_defines.h"
#include "sol_halfmesh.h"
#include "sol_list.h"
#include "sol_math.h"
#include "sol_render.h"

namespace sol {
    // Kept decomposed rather than as a Mat4, so repeated gizmo drags do not
    // accumulate drift in a matrix and an absolute angle stays readable.
    struct Transform {
        Vec3    position;
        Vec3    rotation;   // Euler radians
        Vec3    scale;
    };

    // Scale 1: a zeroed Transform would collapse everything, so use this.
    Transform   TransformDefault();
    Mat4        TransformToMat4( const Transform & transform );

    struct Primitive {
        HalfMesh            halfMesh;
        RenderMaterial      material;
        RenderMeshHandle    renderMesh;
        Transform           transform;
    };

    constexpr i32 kNoPrimitive = -1;
    constexpr Vec4 kSelectionTint = { 1.9f, 1.35f, 0.55f, 1.0f };

    struct World {
        List<Primitive> primitives;
        i32             selected; // kNoPrimitive when nothing is selected.
    };

    World WorldCreate();

    RenderMeshHandle    RenderMeshFromHalfMesh( Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material );
    i32                 WorldAddPrimitive( World & world, Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material, const Transform & transform );
    bool                WorldRebuildPrimitive( World & world, Renderer * r, i32 primitive );
    void                WorldSetPrimitiveTransform( World & world, Renderer * r, i32 primitive, const Transform & transform );
    bool                WorldGetPrimitiveTransform( const World & world, i32 primitive, Transform * outTransform );

    bool                WorldPick( const World & world, Renderer * r, Vec3 rayOrigin, Vec3 rayDirection, i32 * outPrimitive );
    void                WorldSetSelected( World & world, Renderer * r, i32 primitive );
    void                WorldSetPrimitiveHighlight( World & world, Renderer * r, i32 primitive, bool highlight );
    // selectedVertex is drawn in its own colour, or kHMNone for none.
    bool                WorldSetEditOverlay( World & world, Renderer * r, i32 primitive, i32 selectedVertex );

    // Half-mesh vertices addressed in world space, so an editor can hand them
    // to the same gizmo that moves whole objects. Setting one only changes the
    // authored mesh and its face normals: follow it with WorldRebuildPrimitive
    // to see the result, which lets a drag batch many moves into one rebuild.
    bool                WorldGetVertexPosition( const World & world, i32 primitive, i32 vertex, Vec3 * outWorld );
    bool                WorldSetVertexPosition( World & world, i32 primitive, i32 vertex, Vec3 worldPosition );
    bool                WorldRemovePrimitive( World & world, Renderer * r, i32 primitive );
    i32                 WorldRemapPrimitive( i32 held, i32 removed );

    bool                WorldCreateDefaultLevel( World & world, Renderer * r );
    void                WorldFree( World & world );

} // namespace sol
