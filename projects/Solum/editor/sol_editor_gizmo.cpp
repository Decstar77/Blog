#include "sol_editor_gizmo.h"

#include <cmath>

namespace sol {

    // The fragment shader always applies its key light, so gizmo vertices carry
    // the light's own direction as their normal and the axis colours come
    // through flat.
    constexpr Vec3 kGizmoNormal = { 0.41646f, 0.83291f, 0.36440f };

    constexpr Vec3 kGizmoAxisDir[3] = {
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f },
    };

    constexpr Vec3 kGizmoAxisColor[3] = {
        { 0.90f, 0.25f, 0.25f },
        { 0.35f, 0.85f, 0.35f },
        { 0.30f, 0.50f, 0.95f },
    };

    // Perpendicular pair spanning the plane each ring lives in.
    constexpr Vec3 kGizmoAxisU[3] = {
        { 0.0f, 1.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f },
        { 1.0f, 0.0f, 0.0f },
    };

    constexpr Vec3 kGizmoAxisV[3] = {
        { 0.0f, 0.0f, 1.0f },
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
    };

    constexpr Vec4 kGizmoTint = { 1.0f, 1.0f, 1.0f, 1.0f };
    constexpr Vec4 kGizmoTintHot = { 2.0f, 2.0f, 1.4f, 1.0f };

    constexpr i32 kGizmoRingSegments = 48;
    // Fraction of the gizmo's size a ray may miss a handle by and still hit it.
    constexpr f32 kGizmoPickTolerance = 0.12f;
    // Gizmo size as a fraction of its distance from the camera.
    constexpr f32 kGizmoScreenScale = 0.18f;

    static i32 GizmoAxisIndex( GizmoAxis axis ) {
        return (i32)axis - 1;
    }

    static f32 Vec3Component( Vec3 v, i32 index ) {
        return index == 0 ? v.x : ( index == 1 ? v.y : v.z );
    }

    static void Vec3SetComponent( Vec3 * v, i32 index, f32 value ) {
        if( index == 0 )      { v->x = value; }
        else if( index == 1 ) { v->y = value; }
        else                  { v->z = value; }
    }

    static void GizmoPushVertex( List<StaticMeshVertex> & vertices, Vec3 position, Vec3 color ) {
        StaticMeshVertex vertex = {};
        vertex.position = position;
        vertex.normal = kGizmoNormal;
        vertex.color = color;
        vertex.uv = Vec2{ 0.0f, 0.0f };
        ListAdd( vertices, vertex );
    }

    static void GizmoPushLine( List<StaticMeshVertex> & vertices, Vec3 a, Vec3 b, Vec3 color ) {
        GizmoPushVertex( vertices, a, color );
        GizmoPushVertex( vertices, b, color );
    }

    Gizmo GizmoCreate() {
        Gizmo gizmo = {};
        gizmo.mode = GizmoMode_None;
        gizmo.hovered = GizmoAxis_None;
        gizmo.active = GizmoAxis_None;
        gizmo.scale = 1.0f;
        return gizmo;
    }

    void GizmoBuildGeometry( Gizmo & gizmo, List<StaticMeshVertex> & outVertices ) {
        // Arrows first, so ranges 0-2 are translate and 3-5 are rotate.
        for( i32 axis = 0; axis < 3; axis++ ) {
            const i32 first = outVertices.count;
            const Vec3 dir = kGizmoAxisDir[axis];
            const Vec3 color = kGizmoAxisColor[axis];
            const Vec3 tip = dir;

            GizmoPushLine( outVertices, Vec3{ 0.0f, 0.0f, 0.0f }, tip, color );

            // Two barbs, one in each of the plane's directions, so the arrow
            // reads as pointing whichever way it is seen from.
            const Vec3 barbBase = dir * 0.82f;
            const Vec3 u = kGizmoAxisU[axis] * 0.07f;
            const Vec3 v = kGizmoAxisV[axis] * 0.07f;
            GizmoPushLine( outVertices, tip, barbBase + u, color );
            GizmoPushLine( outVertices, tip, barbBase - u, color );
            GizmoPushLine( outVertices, tip, barbBase + v, color );
            GizmoPushLine( outVertices, tip, barbBase - v, color );

            gizmo.ranges[axis].firstVertex = first;
            gizmo.ranges[axis].vertexCount = outVertices.count - first;
            gizmo.ranges[axis].tint = kGizmoTint;
        }

        for( i32 axis = 0; axis < 3; axis++ ) {
            const i32 first = outVertices.count;
            const Vec3 color = kGizmoAxisColor[axis];
            const Vec3 u = kGizmoAxisU[axis];
            const Vec3 v = kGizmoAxisV[axis];

            for( i32 i = 0; i < kGizmoRingSegments; i++ ) {
                const f32 a0 = kTwoPi * (f32)i / (f32)kGizmoRingSegments;
                const f32 a1 = kTwoPi * (f32)( i + 1 ) / (f32)kGizmoRingSegments;
                const Vec3 p0 = u * cosf( a0 ) + v * sinf( a0 );
                const Vec3 p1 = u * cosf( a1 ) + v * sinf( a1 );
                GizmoPushLine( outVertices, p0, p1, color );
            }

            const i32 range = 3 + axis;
            gizmo.ranges[range].firstVertex = first;
            gizmo.ranges[range].vertexCount = outVertices.count - first;
            gizmo.ranges[range].tint = kGizmoTint;
        }
    }

    f32 GizmoScaleFor( Vec3 center, Vec3 cameraPosition ) {
        const f32 distance = Vec3Length( center - cameraPosition );
        // A gizmo on top of the camera would otherwise collapse to nothing.
        return Max( distance * kGizmoScreenScale, 0.05f );
    }

    Mat4 GizmoDrawTransform( const Gizmo & gizmo ) {
        return Mat4Translate( gizmo.center ) *
               Mat4Scale( Vec3{ gizmo.scale, gizmo.scale, gizmo.scale } );
    }

    i32 GizmoDrawRanges( const Gizmo & gizmo, RenderGizmoRange * outRanges ) {
        if( gizmo.mode == GizmoMode_None ) {
            return 0;
        }

        // A drag holds the highlight on its own axis, so the handle stays lit
        // even once the cursor has left it.
        const GizmoAxis lit = gizmo.active != GizmoAxis_None ? gizmo.active : gizmo.hovered;
        const i32 base = gizmo.mode == GizmoMode_Translate ? 0 : 3;

        for( i32 axis = 0; axis < 3; axis++ ) {
            outRanges[axis] = gizmo.ranges[base + axis];
            outRanges[axis].tint = GizmoAxisIndex( lit ) == axis ? kGizmoTintHot : kGizmoTint;
        }
        return 3;
    }

    // Distance from the ray to the segment running from the gizmo's centre out
    // along one axis, and the axis parameter it was closest at.
    static bool GizmoAxisDistance( const Gizmo & gizmo, i32 axis, Vec3 rayOrigin, Vec3 rayDirection,
                                   f32 * outDistance ) {
        f32 lineT = 0.0f;
        if( !RayLineClosest( rayOrigin, rayDirection, gizmo.center, kGizmoAxisDir[axis], &lineT ) ) {
            return false;
        }

        // The handle is a segment, not a line, so a ray passing the axis far
        // beyond the arrowhead is not touching it.
        if( lineT < 0.0f ) { lineT = 0.0f; }
        if( lineT > gizmo.scale ) { lineT = gizmo.scale; }

        const Vec3 onAxis = gizmo.center + kGizmoAxisDir[axis] * lineT;
        const f32 denom = Vec3Dot( rayDirection, rayDirection );
        if( denom <= 0.0f ) {
            return false;
        }

        const f32 rayT = Vec3Dot( onAxis - rayOrigin, rayDirection ) / denom;
        if( rayT < 0.0f ) {
            return false;
        }

        const Vec3 onRay = rayOrigin + rayDirection * rayT;
        *outDistance = Vec3Length( onAxis - onRay );
        return true;
    }

    // How far the ray's meeting with a ring's plane falls from the ring itself.
    static bool GizmoRingDistance( const Gizmo & gizmo, i32 axis, Vec3 rayOrigin, Vec3 rayDirection,
                                   f32 * outDistance ) {
        f32 hitT = 0.0f;
        if( !RayPlaneIntersect( rayOrigin, rayDirection, gizmo.center, kGizmoAxisDir[axis], &hitT ) ) {
            return false;
        }
        if( hitT < 0.0f ) {
            return false;
        }

        const Vec3 hit = rayOrigin + rayDirection * hitT;
        const f32 radius = Vec3Length( hit - gizmo.center );
        *outDistance = radius > gizmo.scale ? radius - gizmo.scale : gizmo.scale - radius;
        return true;
    }

    GizmoAxis GizmoPick( const Gizmo & gizmo, Vec3 rayOrigin, Vec3 rayDirection ) {
        if( gizmo.mode == GizmoMode_None ) {
            return GizmoAxis_None;
        }

        const f32 tolerance = gizmo.scale * kGizmoPickTolerance;
        GizmoAxis best = GizmoAxis_None;
        f32 bestDistance = tolerance;

        for( i32 axis = 0; axis < 3; axis++ ) {
            f32 distance = 0.0f;
            const bool hit = gizmo.mode == GizmoMode_Translate
                ? GizmoAxisDistance( gizmo, axis, rayOrigin, rayDirection, &distance )
                : GizmoRingDistance( gizmo, axis, rayOrigin, rayDirection, &distance );

            if( hit && distance < bestDistance ) {
                bestDistance = distance;
                best = (GizmoAxis)( axis + 1 );
            }
        }

        return best;
    }

    // The angle the ray meets a ring's plane at, measured in that ring's basis.
    static bool GizmoRingAngle( const Gizmo & gizmo, i32 axis, Vec3 rayOrigin, Vec3 rayDirection,
                                f32 * outAngle ) {
        f32 hitT = 0.0f;
        if( !RayPlaneIntersect( rayOrigin, rayDirection, gizmo.center, kGizmoAxisDir[axis], &hitT ) ) {
            return false;
        }

        const Vec3 offset = ( rayOrigin + rayDirection * hitT ) - gizmo.center;
        *outAngle = atan2f( Vec3Dot( offset, kGizmoAxisV[axis] ), Vec3Dot( offset, kGizmoAxisU[axis] ) );
        return true;
    }

    bool GizmoBeginDrag( Gizmo & gizmo, GizmoAxis axis, const Transform & transform,
                         Vec3 rayOrigin, Vec3 rayDirection ) {
        if( axis == GizmoAxis_None || gizmo.mode == GizmoMode_None ) {
            return false;
        }

        const i32 index = GizmoAxisIndex( axis );
        if( gizmo.mode == GizmoMode_Translate ) {
            f32 lineT = 0.0f;
            if( !RayLineClosest( rayOrigin, rayDirection, gizmo.center, kGizmoAxisDir[index], &lineT ) ) {
                return false;
            }
            gizmo.startAxisT = lineT;
        } else {
            f32 angle = 0.0f;
            if( !GizmoRingAngle( gizmo, index, rayOrigin, rayDirection, &angle ) ) {
                return false;
            }
            gizmo.startAngle = angle;
        }

        gizmo.active = axis;
        gizmo.startTransform = transform;
        return true;
    }

    bool GizmoUpdateDrag( const Gizmo & gizmo, Vec3 rayOrigin, Vec3 rayDirection, f32 snapStep,
                          f32 rotateSnapStep, Transform * outTransform ) {
        if( gizmo.active == GizmoAxis_None ) {
            return false;
        }

        const i32 index = GizmoAxisIndex( gizmo.active );
        Transform result = gizmo.startTransform;

        if( gizmo.mode == GizmoMode_Translate ) {
            f32 lineT = 0.0f;
            if( !RayLineClosest( rayOrigin, rayDirection, gizmo.center, kGizmoAxisDir[index], &lineT ) ) {
                return false;
            }

            const f32 moved = Vec3Component( gizmo.startTransform.position, index ) +
                              ( lineT - gizmo.startAxisT );
            // Snapped to the grid itself rather than to the distance dragged,
            // so an object lands on the same lines the geometry does.
            Vec3SetComponent( &result.position, index, SnapTo( moved, snapStep ) );
        } else {
            f32 angle = 0.0f;
            if( !GizmoRingAngle( gizmo, index, rayOrigin, rayDirection, &angle ) ) {
                return false;
            }

            const f32 turned = Vec3Component( gizmo.startTransform.rotation, index ) +
                               ( angle - gizmo.startAngle );
            // Snapped to the absolute angle rather than to the amount turned,
            // so a snapped drag always leaves the object on a multiple of the
            // step even if it did not start on one.
            Vec3SetComponent( &result.rotation, index, SnapTo( turned, rotateSnapStep ) );
        }

        *outTransform = result;
        return true;
    }

    void GizmoEndDrag( Gizmo & gizmo ) {
        gizmo.active = GizmoAxis_None;
    }

} // namespace sol
