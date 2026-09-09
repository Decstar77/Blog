#include "nerf_math.h"

#include <cmath>

namespace nerf {

    Vec3 operator+( Vec3 a, Vec3 b ) {
        return Vec3{ a.x + b.x, a.y + b.y, a.z + b.z };
    }

    Vec3 operator-( Vec3 a, Vec3 b ) {
        return Vec3{ a.x - b.x, a.y - b.y, a.z - b.z };
    }

    Vec3 operator*( Vec3 v, f32 s ) {
        return Vec3{ v.x * s, v.y * s, v.z * s };
    }

    Vec3 operator*( f32 s, Vec3 v ) {
        return v * s;
    }

    f32 Vec3Dot( Vec3 a, Vec3 b ) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    Vec3 Vec3Cross( Vec3 a, Vec3 b ) {
        return Vec3{
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        };
    }

    f32 Vec3Length( Vec3 v ) {
        return sqrtf( Vec3Dot( v, v ) );
    }

    Vec3 Vec3Normalize( Vec3 v ) {
        const f32 len = Vec3Length( v );
        if( len <= 0.0f ) {
            return Vec3{ 0.0f, 0.0f, 0.0f };
        }
        return v * ( 1.0f / len );
    }

    Mat4 Mat4Identity() {
        Mat4 r = {};
        r.m[0][0] = 1.0f;
        r.m[1][1] = 1.0f;
        r.m[2][2] = 1.0f;
        r.m[3][3] = 1.0f;
        return r;
    }

    Mat4 operator*( const Mat4 & a, const Mat4 & b ) {
        Mat4 r = {};
        for( i32 row = 0; row < 4; row++ ) {
            for( i32 col = 0; col < 4; col++ ) {
                f32 sum = 0.0f;
                for( i32 k = 0; k < 4; k++ ) {
                    sum += a.m[row][k] * b.m[k][col];
                }
                r.m[row][col] = sum;
            }
        }
        return r;
    }

    Vec4 operator*( const Mat4 & a, Vec4 v ) {
        return Vec4{
            a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z + a.m[0][3] * v.w,
            a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z + a.m[1][3] * v.w,
            a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z + a.m[2][3] * v.w,
            a.m[3][0] * v.x + a.m[3][1] * v.y + a.m[3][2] * v.z + a.m[3][3] * v.w,
        };
    }

    Vec3 Mat4MulPoint( const Mat4 & a, Vec3 p ) {
        const Vec4 r = a * Vec4{ p.x, p.y, p.z, 1.0f };
        return Vec3{ r.x, r.y, r.z };
    }

    Vec3 Mat4MulDir( const Mat4 & a, Vec3 d ) {
        const Vec4 r = a * Vec4{ d.x, d.y, d.z, 0.0f };
        return Vec3{ r.x, r.y, r.z };
    }

    Vec3 Mat4Translation( const Mat4 & a ) {
        return Vec3{ a.m[0][3], a.m[1][3], a.m[2][3] };
    }

    Vec3 Mat4Right( const Mat4 & a ) {
        return Vec3{ a.m[0][0], a.m[1][0], a.m[2][0] };
    }

    Vec3 Mat4Up( const Mat4 & a ) {
        return Vec3{ a.m[0][1], a.m[1][1], a.m[2][1] };
    }

    Vec3 Mat4Forward( const Mat4 & a ) {
        return Vec3{ -a.m[0][2], -a.m[1][2], -a.m[2][2] };
    }

    f32 FocalLengthFromFovX( f32 fovX, i32 imageWidth ) {
        return 0.5f * (f32)imageWidth / tanf( 0.5f * fovX );
    }

    RandomSeries RandomSeed( u32 seed ) {
        RandomSeries rng = {};
        rng.state = seed != 0 ? seed : 0x9e3779b9u;
        return rng;
    }

    u32 RandomNextU32( RandomSeries * rng ) {
        u32 x = rng->state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        rng->state = x;
        return x;
    }

    i32 RandomBelow( RandomSeries * rng, i32 bound ) {
        if( bound <= 1 ) {
            return 0;
        }
        // Modulo bias is on the order of bound / 2^32 here, far below anything that matters for
        // picking pixels, so it is not worth a rejection loop.
        return (i32)( RandomNextU32( rng ) % (u32)bound );
    }

    f32 RandomUnilateral( RandomSeries * rng ) {
        // 24 bits, the most a f32 can hold exactly.
        return (f32)( RandomNextU32( rng ) >> 8 ) * ( 1.0f / 16777216.0f );
    }

    f32 RandomBilateral( RandomSeries * rng ) {
        return RandomUnilateral( rng ) * 2.0f - 1.0f;
    }
}
