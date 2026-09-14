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

    // Three arrows then three rings, in X, Y, Z order.
    constexpr i32 kGizmoRangeCount = 6;

    struct Gizmo {
        GizmoMode           mode;
        GizmoAxis           hovered;
        // GizmoAxis_None unless a drag is running.
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

    Gizmo GizmoCreate();

    // Local-space line geometry, uploaded once. Fills gizmo.ranges with the
    // slice of it each handle occupies.
    void GizmoBuildGeometry( Gizmo & gizmo, List<StaticMeshVertex> & outVertices );

    // World size that keeps the gizmo roughly constant on screen.
    f32 GizmoScaleFor( Vec3 center, Vec3 cameraPosition );

    // Nearest handle the ray touches, or GizmoAxis_None.
    GizmoAxis GizmoPick( const Gizmo & gizmo, Vec3 rayOrigin, Vec3 rayDirection );

    bool GizmoBeginDrag( Gizmo & gizmo, GizmoAxis axis, const Transform & transform,
                         Vec3 rayOrigin, Vec3 rayDirection );

    // The subject's transform for this cursor ray. snapStep <= 0 disables
    // translation snapping; rotation is never snapped.
    bool GizmoUpdateDrag( const Gizmo & gizmo, Vec3 rayOrigin, Vec3 rayDirection, f32 snapStep,
                          Transform * outTransform );

    void GizmoEndDrag( Gizmo & gizmo );

    // Which slices to draw this frame and how to tint them. Returns the count.
    i32 GizmoDrawRanges( const Gizmo & gizmo, RenderGizmoRange * outRanges );

    // Where to draw them: the gizmo sits on the subject at its own world size.
    Mat4 GizmoDrawTransform( const Gizmo & gizmo );

} // namespace sol
