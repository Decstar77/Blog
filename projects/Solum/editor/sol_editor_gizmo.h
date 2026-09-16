#pragma once

#include "sol_math.h"
#include "sol_render.h"
#include "sol_world.h"

namespace sol {

    enum GizmoMode {
        GizmoMode_None,
        GizmoMode_Translate,
        GizmoMode_Rotate,
    };

    enum GizmoAxis {
        GizmoAxis_None,
        GizmoAxis_X,
        GizmoAxis_Y,
        GizmoAxis_Z,
    };

    constexpr i32 kGizmoRangeCount = 6;

    struct Gizmo {
        GizmoMode           mode;
        GizmoAxis           hovered;
        GizmoAxis           active;

        // Where the gizmo was last drawn, and the world size it was drawn at.
        // Picking uses these, so they have to be refreshed before a press is
        // tested against them.
        Vec3                center;
        f32                 scale;

        // The subject as it was when the drag began. Every update is computed
        // from this rather than from the previous frame, so a drag that passes
        // back over its start returns the object exactly where it was.
        Transform           startTransform;
        f32                 startAxisT;
        f32                 startAngle;

        RenderGizmoRange    ranges[kGizmoRangeCount];
    };

    Gizmo       GizmoCreate();
    void        GizmoBuildGeometry( Gizmo & gizmo, List<StaticMeshVertex> & outVertices );
    f32         GizmoScaleFor( Vec3 center, Vec3 cameraPosition );
    GizmoAxis   GizmoPick( const Gizmo & gizmo, Vec3 rayOrigin, Vec3 rayDirection );
    bool        GizmoBeginDrag( Gizmo & gizmo, GizmoAxis axis, const Transform & transform, Vec3 rayOrigin, Vec3 rayDirection );
    bool        GizmoUpdateDrag( const Gizmo & gizmo, Vec3 rayOrigin, Vec3 rayDirection, f32 snapStep, Transform * outTransform );
    void        GizmoEndDrag( Gizmo & gizmo );
    i32         GizmoDrawRanges( const Gizmo & gizmo, RenderGizmoRange * outRanges );
    Mat4        GizmoDrawTransform( const Gizmo & gizmo );

} // namespace sol
