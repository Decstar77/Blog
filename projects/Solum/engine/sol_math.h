#pragma once
#include "sol_defines.h"

namespace sol {
    struct Vec2 {
        f32 x;
        f32 y;
    };

    struct Vec3 {
        f32 x;
        f32 y;
        f32 z;
    };

    struct Vec4 {
        f32 x;
        f32 y;
        f32 z;
        f32 w;
    };

    struct Mat3 {
        f32 m[3][3];
    };

    // Row-major 4x4
    struct Mat4 {
        f32 m[4][4];
    };

    Vec3    operator+( Vec3 a, Vec3 b );
    Vec3    operator-( Vec3 a, Vec3 b );
    Vec3    operator*( Vec3 v, f32 s );
    Vec3    operator*( f32 s, Vec3 v );
    f32     Vec3Dot( Vec3 a, Vec3 b );
    Vec3    Vec3Cross( Vec3 a, Vec3 b );
    f32     Vec3Length( Vec3 v );
    Vec3    Vec3Normalize( Vec3 v );

    Mat4    Mat4Identity();
    Mat4    Mat4Translate( Vec3 translation );
    Mat4    Mat4Scale( Vec3 scale );

    Mat4    Mat4RotateX( f32 radians );
    Mat4    Mat4RotateY( f32 radians );
    Mat4    Mat4RotateZ( f32 radians );
    // Ry * Rx * Rz, matching the camera's yaw-then-pitch convention.
    Mat4    Mat4FromEuler( Vec3 radians );

    // Nearest multiple of step. step <= 0 passes the value straight through, so
    // callers do not have to guard a disabled snap.
    f32     SnapTo( f32 value, f32 step );
    Vec3    Vec3SnapTo( Vec3 value, f32 step );

    // World to view. Right-handed: the camera looks down its own -z.
    Mat4    Mat4LookAt( Vec3 eye, Vec3 target, Vec3 up );

    // View to clip, with depth in [0, 1] the way Vulkan wants it. Y is NOT
    // flipped here - the renderer draws with a negative viewport height, which
    // already puts +y up. Flipping in both places would cancel out.
    Mat4    Mat4Perspective( f32 fovY, f32 aspect, f32 nearZ, f32 farZ );

    // Same conventions as Mat4Perspective: depth lands in [0, 1] and y is NOT
    // flipped here. Extents are half-sizes measured from the centre of the
    // view, so the visible box is 2*halfWidth by 2*halfHeight.
    Mat4    Mat4Orthographic( f32 halfWidth, f32 halfHeight, f32 nearZ, f32 farZ );

    Mat4    operator*( const Mat4 & a, const Mat4 & b );
    Vec4    operator*( const Mat4 & a, Vec4 v );
    Vec3    Mat4MulPoint( const Mat4 & a, Vec3 p );
    Vec3    Mat4MulDir( const Mat4 & a, Vec3 d );

    // Camera-to-world basis. NeRF synthetic uses the OpenGL/Blender convention:
    // +x right, +y up, -z forward (the camera looks down its own -z axis).
    Vec3    Mat4Translation( const Mat4 & a );
    Vec3    Mat4Right( const Mat4 & a );
    Vec3    Mat4Up( const Mat4 & a );
    Vec3    Mat4Forward( const Mat4 & a );

    f32     FocalLengthFromFovX( f32 fovX, i32 imageWidth );

    // Slab test. outDistance is how far along direction the ray first meets the
    // box, and is zero when the ray starts inside it. direction need not be
    // normalised, but the distance comes back in units of it if it is not.
    bool    RayAabbIntersect( Vec3 origin, Vec3 direction, Vec3 boundsMin, Vec3 boundsMax,
                              f32 * outDistance );

    // Point on the infinite line through linePoint that is nearest the ray.
    // outLineT is the parameter along lineDir. False when the two are parallel.
    bool    RayLineClosest( Vec3 rayOrigin, Vec3 rayDirection, Vec3 linePoint, Vec3 lineDirection,
                            f32 * outLineT );

    // Distance along the ray to the plane. False when the ray runs parallel to it.
    bool    RayPlaneIntersect( Vec3 rayOrigin, Vec3 rayDirection, Vec3 planePoint, Vec3 planeNormal,
                               f32 * outDistance );

    // Moller-Trumbore, two sided: a back face hits as readily as a front one,
    // because the editor has to be able to pick the inside of a box it is
    // standing in. outDistance is along direction and is never negative - a
    // triangle behind the origin is a miss, not a negative hit.
    bool    RayTriangleIntersect( Vec3 rayOrigin, Vec3 rayDirection, Vec3 a, Vec3 b, Vec3 c,
                                  f32 * outDistance );

    // xorshift32. Seeded explicitly so a given seed always replays the same stream, and 0 is
    // folded to a non zero constant because a zero state gets stuck at zero.
    struct RandomSeries {
        u32 state;
    };

    RandomSeries    RandomSeed( u32 seed );
    u32             RandomNextU32( RandomSeries * rng );
    i32             RandomBelow( RandomSeries * rng, i32 bound );  // uniform in [ 0, bound )
    f32             RandomUnilateral( RandomSeries * rng );        // uniform in [ 0, 1 )
    f32             RandomBilateral( RandomSeries * rng );         // uniform in [ -1, 1 )
}
