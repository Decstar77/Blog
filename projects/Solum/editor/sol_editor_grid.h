#pragma once

#include "sol_math.h"

#include <cmath>

namespace sol {
    // The plane authoring happens on. Everything the editor places - a dragged
    // out plane, a built box - is expressed in this plane's own coordinates and
    // only turned back into world space at the end, so none of that code has to
    // know which way the grid is facing.
    struct EditorGrid {
        f32  step;      // world units between lines, and the snap increment
        Vec3 normal;    // unit, the plane's up
        Vec3 position;  // a point on the plane, and the origin snapping counts from
    };

    // Matches the renderer's built-in grid: one unit cells on y = 0.
    inline EditorGrid EditorGridDefault() {
        EditorGrid grid = {};
        grid.step = 1.0f;
        grid.normal = Vec3{ 0.0f, 1.0f, 0.0f };
        grid.position = Vec3{ 0.0f, 0.0f, 0.0f };
        return grid;
    }

    // Yaw and pitch that carry local +y onto the grid normal, in the engine's
    // Ry * Rx * Rz order. Both the basis below and the transform handed to a
    // primitive come from these two angles, which is what keeps a box's local
    // axes and the axes it was sized along the same axes.
    inline void EditorGridAngles( const EditorGrid & grid, f32 * outYaw, f32 * outPitch ) {
        const Vec3 n = Vec3Normalize( grid.normal );
        const f32 clamped = n.y < -1.0f ? -1.0f : ( n.y > 1.0f ? 1.0f : n.y );
        // atan2( 0, 0 ) is zero, so a straight up or down normal falls out as
        // yaw 0 rather than as a NaN.
        *outYaw = atan2f( n.x, n.z );
        *outPitch = acosf( clamped );
    }

    // Euler angles for a primitive whose local space should sit on the grid:
    // local +y along the normal, local +x and +z along the two in-plane axes.
    inline Vec3 EditorGridRotation( const EditorGrid & grid ) {
        f32 yaw = 0.0f;
        f32 pitch = 0.0f;
        EditorGridAngles( grid, &yaw, &pitch );
        return Vec3{ pitch, yaw, 0.0f };
    }

    // Grid local space to world, which is what the renderer draws its grid
    // with. The renderer builds that grid on the xz plane around its own
    // origin - exactly the space EditorGridToWorld maps from - so this one
    // matrix is what keeps the lines on screen and the lines geometry snaps to
    // the same lines.
    inline Mat4 EditorGridTransform( const EditorGrid & grid ) {
        return Mat4Translate( grid.position ) * Mat4FromEuler( EditorGridRotation( grid ) );
    }

    // The two in-plane axes, u along local +x and v along local +z. Derived
    // from the same angles as EditorGridRotation rather than from a cross
    // product, so there is no seam where a cross product would have degenerated.
    inline void EditorGridBasis( const EditorGrid & grid, Vec3 * outU, Vec3 * outV ) {
        f32 yaw = 0.0f;
        f32 pitch = 0.0f;
        EditorGridAngles( grid, &yaw, &pitch );
        const f32 cy = cosf( yaw );
        const f32 sy = sinf( yaw );
        const f32 cp = cosf( pitch );
        const f32 sp = sinf( pitch );

        *outU = Vec3{ cy, 0.0f, -sy };
        *outV = Vec3{ sy * cp, -sp, cy * cp };
    }

    // The normal as the rotation actually produces it, which is the grid's own
    // normal re-normalised. Use this rather than grid.normal wherever it is
    // going to be scaled by a height, so a sloppily authored normal cannot
    // stretch the result.
    inline Vec3 EditorGridUp( const EditorGrid & grid ) {
        return Vec3Normalize( grid.normal );
    }

    // World point to grid coordinates. z is the signed height off the plane, so
    // a point already on it comes back with z = 0.
    inline Vec3 EditorGridToLocal( const EditorGrid & grid, Vec3 world ) {
        Vec3 u = {};
        Vec3 v = {};
        EditorGridBasis( grid, &u, &v );
        const Vec3 d = world - grid.position;
        return Vec3{ Vec3Dot( d, u ), Vec3Dot( d, v ), Vec3Dot( d, EditorGridUp( grid ) ) };
    }

    // Grid coordinates back to a world point, with local.z lifting it off the
    // plane along the normal.
    inline Vec3 EditorGridToWorld( const EditorGrid & grid, Vec3 local ) {
        Vec3 u = {};
        Vec3 v = {};
        EditorGridBasis( grid, &u, &v );
        return grid.position + u * local.x + v * local.y + EditorGridUp( grid ) * local.z;
    }

    // Snaps in the plane's own axes, not in world x/y/z, which is the whole
    // point: a tilted grid still lands geometry on its own lines.
    inline Vec3 EditorGridSnapLocal( const EditorGrid & grid, Vec3 local ) {
        return Vec3{ SnapTo( local.x, grid.step ), SnapTo( local.y, grid.step ), SnapTo( local.z, grid.step ) };
    }

    inline Vec3 EditorGridSnapWorld( const EditorGrid & grid, Vec3 world ) {
        return EditorGridToWorld( grid, EditorGridSnapLocal( grid, EditorGridToLocal( grid, world ) ) );
    }

    // Where a ray crosses the grid, in grid coordinates (z is always 0). False
    // when the ray runs along the plane or only meets it behind the camera,
    // which is what an orthographic pane looking edge-on at the grid produces.
    inline bool EditorGridRaycast( const EditorGrid & grid, Vec3 rayOrigin, Vec3 rayDirection, Vec3 * outLocal ) {
        f32 distance = 0.0f;
        if( !RayPlaneIntersect( rayOrigin, rayDirection, grid.position, EditorGridUp( grid ), &distance ) ) {
            return false;
        }
        if( distance < 0.0f ) {
            return false;
        }

        Vec3 local = EditorGridToLocal( grid, rayOrigin + rayDirection * distance );
        // The solve puts it on the plane to within float error; this says so
        // exactly, so callers can treat z as a height they own.
        local.z = 0.0f;
        *outLocal = local;
        return true;
    }

    // Height along the normal of the point on the line through the grid at
    // planePoint that is nearest the ray. This is how a drag that is no longer
    // touching the grid - an extrude pulling straight up off it - still reads a
    // distance. False when the ray is parallel to the normal, where a screen
    // looking straight down the extrusion axis has no height to give.
    inline bool EditorGridRaycastHeight( const EditorGrid & grid, Vec3 planePoint, Vec3 rayOrigin, Vec3 rayDirection, f32 * outHeight ) {
        return RayLineClosest( rayOrigin, rayDirection, planePoint, EditorGridUp( grid ), outHeight );
    }

} // namespace sol
