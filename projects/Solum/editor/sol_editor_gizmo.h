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

    // Rotation snaps to 15 degree steps unless the caller passes 0 to disable
    // it, which is what holding Ctrl during a drag does.
    constexpr f32 kGizmoRotateSnap = 15.0f * 3.14159265358979323846f / 180.0f;

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
    // snapStep is the grid step a translate lands on; rotateSnapStep is the angle
    // in radians a rotation lands on. Either at 0 passes the raw value through.
    bool        GizmoUpdateDrag( const Gizmo & gizmo, Vec3 rayOrigin, Vec3 rayDirection, f32 snapStep, f32 rotateSnapStep, Transform * outTransform );
    void        GizmoEndDrag( Gizmo & gizmo );
    i32         GizmoDrawRanges( const Gizmo & gizmo, RenderGizmoRange * outRanges );
    Mat4        GizmoDrawTransform( const Gizmo & gizmo );

} // namespace sol
