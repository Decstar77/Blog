#pragma once
#include "nerf_defines.h"

namespace nerf {
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

    // Row-major 4x4, laid out exactly like the nested arrays in transforms_*.json.
    // m[ row ][ col ], so m[ 0 ][ 3 ] is the x component of the translation.
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

}
