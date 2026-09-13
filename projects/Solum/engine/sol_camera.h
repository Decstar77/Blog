#pragma once
#include "sol_defines.h"
#include "sol_math.h"

namespace sol {

    struct FlyCameraInput {
        bool    forward;
        bool    back;
        bool    left;
        bool    right;
        bool    up;
        bool    down;
        bool    fast;

        // Mouse travel in pixels since the last update, applied only while  looking. A shell that is not in look mode leaves these at zero.
        bool    looking;
        f32     lookDeltaX;
        f32     lookDeltaY;
    };

    struct FlyCamera {
        Vec3    position;
        // Radians. yaw 0 / pitch 0 looks down -z, matching the math convention.
        f32     yaw;
        f32     pitch;
        f32     moveSpeed;      // units per second
        f32     lookSpeed;      // radians per pixel of mouse travel
    };

    FlyCamera   FlyCameraDefault();

    Vec3        FlyCameraForward( const FlyCamera & camera );
    void        FlyCameraUpdate( FlyCamera * camera, const FlyCameraInput & input, f32 dt );
    Mat4        FlyCameraViewProjection( const FlyCamera & camera, i32 width, i32 height );

} // namespace sol
