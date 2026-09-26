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
    // The camera's basis in world space, matching the one its projection and
    // picking ray are built from.
    void        FlyCameraAxes( const FlyCamera & camera, Vec3 * outRight, Vec3 * outUp, Vec3 * outForward );
    void        FlyCameraUpdate( FlyCamera * camera, const FlyCameraInput & input, f32 dt );
    Mat4        FlyCameraViewProjection( const FlyCamera & camera, i32 width, i32 height );
    void        FlyCameraScreenRay( const FlyCamera & camera, f32 pixelX, f32 pixelY, i32 pixelWidth, i32 pixelHeight, Vec3 * outOrigin, Vec3 * outDirection );

    enum OrthoAxis {
        OrthoAxis_Top,      // down -y, +x right and -z up on screen
        OrthoAxis_Front,    // down -z, +x right and +y up
        OrthoAxis_Side,     // down -x, +z right and +y up
    };

    struct OrthoCamera {
        OrthoAxis   axis;
        Vec3        center;
        f32         halfHeight;
        f32         zoomSpeed;
    };

    struct OrthoCameraInput {
        bool    panning;
        f32     panDeltaX;      // pixels of drag since the last update
        f32     panDeltaY;
        f32     zoomTicks;      // wheel notches, positive zooms in
    };

    OrthoCamera OrthoCameraDefault( OrthoAxis axis );
    // Screen right, screen up and view direction in world space. Each is
    // exactly a world axis, which is what lets a 2D pane snap in world axes.
    void        OrthoCameraAxes( const OrthoCamera & camera, Vec3 * outRight, Vec3 * outUp, Vec3 * outForward );
    void        OrthoCameraUpdate( OrthoCamera * camera, const OrthoCameraInput & input, i32 pixelHeight );
    Mat4        OrthoCameraViewProjection( const OrthoCamera & camera, i32 width, i32 height );
    Vec3        OrthoCameraScreenToWorld( const OrthoCamera & camera, f32 pixelX, f32 pixelY, i32 pixelWidth, i32 pixelHeight );
    void        OrthoCameraScreenRay( const OrthoCamera & camera, f32 pixelX, f32 pixelY, i32 pixelWidth, i32 pixelHeight, Vec3 * outOrigin, Vec3 * outDirection );

} // namespace sol
