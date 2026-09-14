#include "sol_math.h"

#include <cmath>

namespace sol {

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

    Mat4 Mat4Translate( Vec3 translation ) {
        // Row-major, so the translation is the last column.
        Mat4 r = Mat4Identity();
        r.m[0][3] = translation.x;
        r.m[1][3] = translation.y;
        r.m[2][3] = translation.z;
        return r;
    }

    Mat4 Mat4Scale( Vec3 scale ) {
        Mat4 r = Mat4Identity();
        r.m[0][0] = scale.x;
        r.m[1][1] = scale.y;
        r.m[2][2] = scale.z;
        return r;
    }

    Mat4 Mat4RotateX( f32 radians ) {
        const f32 c = cosf( radians );
        const f32 s = sinf( radians );
        Mat4 r = Mat4Identity();
        r.m[1][1] = c;  r.m[1][2] = -s;
        r.m[2][1] = s;  r.m[2][2] = c;
        return r;
    }

    Mat4 Mat4RotateY( f32 radians ) {
        const f32 c = cosf( radians );
        const f32 s = sinf( radians );
        Mat4 r = Mat4Identity();
        r.m[0][0] = c;  r.m[0][2] = s;
        r.m[2][0] = -s; r.m[2][2] = c;
        return r;
    }

    Mat4 Mat4RotateZ( f32 radians ) {
        const f32 c = cosf( radians );
        const f32 s = sinf( radians );
        Mat4 r = Mat4Identity();
        r.m[0][0] = c;  r.m[0][1] = -s;
        r.m[1][0] = s;  r.m[1][1] = c;
        return r;
    }

    Mat4 Mat4FromEuler( Vec3 radians ) {
        return Mat4RotateY( radians.y ) * Mat4RotateX( radians.x ) * Mat4RotateZ( radians.z );
    }

    f32 SnapTo( f32 value, f32 step ) {
        if( step <= 0.0f ) {
            return value;
        }
        return roundf( value / step ) * step;
    }

    Vec3 Vec3SnapTo( Vec3 value, f32 step ) {
        return Vec3{ SnapTo( value.x, step ), SnapTo( value.y, step ), SnapTo( value.z, step ) };
    }

    Mat4 Mat4LookAt( Vec3 eye, Vec3 target, Vec3 up ) {
        const Vec3 f = Vec3Normalize( target - eye );
        const Vec3 s = Vec3Normalize( Vec3Cross( f, up ) );
        const Vec3 u = Vec3Cross( s, f );

        // The basis goes in as rows because this is the inverse of the camera's
        // own transform, and the translation is the eye projected onto it.
        Mat4 r = Mat4Identity();
        r.m[0][0] = s.x;    r.m[0][1] = s.y;    r.m[0][2] = s.z;    r.m[0][3] = -Vec3Dot( s, eye );
        r.m[1][0] = u.x;    r.m[1][1] = u.y;    r.m[1][2] = u.z;    r.m[1][3] = -Vec3Dot( u, eye );
        r.m[2][0] = -f.x;   r.m[2][1] = -f.y;   r.m[2][2] = -f.z;   r.m[2][3] = Vec3Dot( f, eye );
        return r;
    }

    Mat4 Mat4Perspective( f32 fovY, f32 aspect, f32 nearZ, f32 farZ ) {
        const f32 t = tanf( 0.5f * fovY );

        Mat4 r = {};
        r.m[0][0] = 1.0f / ( aspect * t );
        r.m[1][1] = 1.0f / t;
        r.m[2][2] = farZ / ( nearZ - farZ );
        r.m[2][3] = ( nearZ * farZ ) / ( nearZ - farZ );
        // Perspective divide by -z, so w carries the view-space depth.
        r.m[3][2] = -1.0f;
        return r;
    }

    Mat4 Mat4Orthographic( f32 halfWidth, f32 halfHeight, f32 nearZ, f32 farZ ) {
        Mat4 r = {};
        r.m[0][0] = 1.0f / halfWidth;
        r.m[1][1] = 1.0f / halfHeight;
        // Solved against the same depth mapping Mat4Perspective uses: a view
        // space z of -nearZ has to come out 0 and -farZ has to come out 1.
        // There is no perspective divide here, so w stays 1.
        r.m[2][2] = 1.0f / ( nearZ - farZ );
        r.m[2][3] = nearZ / ( nearZ - farZ );
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

    bool RayAabbIntersect( Vec3 origin, Vec3 direction, Vec3 boundsMin, Vec3 boundsMax,
                           f32 * outDistance ) {
        f32 tMin = 0.0f;
        f32 tMax = 3.4e38f;

        const f32 o[3] = { origin.x, origin.y, origin.z };
        const f32 d[3] = { direction.x, direction.y, direction.z };
        const f32 lo[3] = { boundsMin.x, boundsMin.y, boundsMin.z };
        const f32 hi[3] = { boundsMax.x, boundsMax.y, boundsMax.z };

        for( i32 axis = 0; axis < 3; axis++ ) {
            if( d[axis] > -1e-8f && d[axis] < 1e-8f ) {
                // Parallel to this pair of planes, so it either started between
                // them or it never crosses them at all.
                if( o[axis] < lo[axis] || o[axis] > hi[axis] ) {
                    return false;
                }
                continue;
            }

            const f32 inverse = 1.0f / d[axis];
            f32 near = ( lo[axis] - o[axis] ) * inverse;
            f32 far = ( hi[axis] - o[axis] ) * inverse;
            if( near > far ) {
                const f32 swap = near;
                near = far;
                far = swap;
            }

            if( near > tMin ) { tMin = near; }
            if( far < tMax )  { tMax = far; }
            if( tMin > tMax ) {
                return false;
            }
        }

        if( outDistance != nullptr ) {
            *outDistance = tMin;
        }
        return true;
    }

    bool RayLineClosest( Vec3 rayOrigin, Vec3 rayDirection, Vec3 linePoint, Vec3 lineDirection,
                         f32 * outLineT ) {
        const Vec3 w = rayOrigin - linePoint;
        const f32 a = Vec3Dot( rayDirection, rayDirection );
        const f32 b = Vec3Dot( rayDirection, lineDirection );
        const f32 c = Vec3Dot( lineDirection, lineDirection );
        const f32 d = Vec3Dot( rayDirection, w );
        const f32 e = Vec3Dot( lineDirection, w );

        // Zero when the two are parallel, where there is no single nearest point.
        const f32 denom = a * c - b * b;
        if( denom > -1e-8f && denom < 1e-8f ) {
            return false;
        }

        if( outLineT != nullptr ) {
            *outLineT = ( a * e - b * d ) / denom;
        }
        return true;
    }

    bool RayPlaneIntersect( Vec3 rayOrigin, Vec3 rayDirection, Vec3 planePoint, Vec3 planeNormal,
                            f32 * outDistance ) {
        const f32 denom = Vec3Dot( rayDirection, planeNormal );
        if( denom > -1e-8f && denom < 1e-8f ) {
            return false;
        }

        if( outDistance != nullptr ) {
            *outDistance = Vec3Dot( planePoint - rayOrigin, planeNormal ) / denom;
        }
        return true;
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
