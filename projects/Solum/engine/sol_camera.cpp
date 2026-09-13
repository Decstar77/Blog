#include "sol_camera.h"

#include <cmath>

namespace sol {

    constexpr f32 kPitchLimit = 89.0f * kDeg2Rad;

    FlyCamera FlyCameraDefault() {
        FlyCamera camera = {};
        // Backed off down +z so geometry at the origin is in view.
        camera.position = Vec3{ 0.0f, 0.0f, 2.0f };
        camera.moveSpeed = 3.0f;
        camera.lookSpeed = 0.0025f;
        return camera;
    }

    Vec3 FlyCameraForward( const FlyCamera & camera ) {
        const f32 cosPitch = cosf( camera.pitch );
        return Vec3{
            cosPitch * sinf( camera.yaw ),
            sinf( camera.pitch ),
            -cosPitch * cosf( camera.yaw ),
        };
    }

    void FlyCameraUpdate( FlyCamera * camera, const FlyCameraInput & input, f32 dt ) {
        if( input.looking ) {
            camera->yaw += input.lookDeltaX * camera->lookSpeed;
            camera->pitch -= input.lookDeltaY * camera->lookSpeed;

            // Stop short of straight up, where the forward vector and world up
            // line up and the right vector collapses.
            if( camera->pitch > kPitchLimit )  { camera->pitch = kPitchLimit; }
            if( camera->pitch < -kPitchLimit ) { camera->pitch = -kPitchLimit; }
        }

        const Vec3 forward = FlyCameraForward( *camera );
        const Vec3 right = Vec3Normalize( Vec3Cross( forward, Vec3{ 0.0f, 1.0f, 0.0f } ) );

        Vec3 move = {};
        if( input.forward ) { move = move + forward; }
        if( input.back )    { move = move - forward; }
        if( input.right )   { move = move + right; }
        if( input.left )    { move = move - right; }
        if( input.up )      { move.y += 1.0f; }
        if( input.down )    { move.y -= 1.0f; }

        f32 speed = camera->moveSpeed;
        if( input.fast ) {
            speed *= 4.0f;
        }

        // Normalised so diagonals are not faster than the axes.
        if( Vec3Length( move ) > 0.0f ) {
            camera->position = camera->position + Vec3Normalize( move ) * ( speed * dt );
        }
    }

    // Zoom is clamped so a runaway wheel cannot collapse the view to nothing or
    // push it out past the depth range.
    constexpr f32 kMinHalfHeight = 0.05f;
    constexpr f32 kMaxHalfHeight = 500.0f;
    // How far back along the view axis the eye sits. The depth range below is
    // twice this, so geometry either side of the centre plane stays visible.
    constexpr f32 kOrthoPullback = 500.0f;

    static void OrthoAxisBasis( OrthoAxis axis, Vec3 * outForward, Vec3 * outUp ) {
        switch( axis ) {
            case OrthoAxis_Front:
                *outForward = Vec3{ 0.0f, 0.0f, -1.0f };
                *outUp = Vec3{ 0.0f, 1.0f, 0.0f };
                break;
            case OrthoAxis_Side:
                *outForward = Vec3{ -1.0f, 0.0f, 0.0f };
                *outUp = Vec3{ 0.0f, 1.0f, 0.0f };
                break;
            case OrthoAxis_Top:
            default:
                *outForward = Vec3{ 0.0f, -1.0f, 0.0f };
                *outUp = Vec3{ 0.0f, 0.0f, -1.0f };
                break;
        }
    }

    OrthoCamera OrthoCameraDefault( OrthoAxis axis ) {
        OrthoCamera camera = {};
        camera.axis = axis;
        camera.center = Vec3{ 0.0f, 0.0f, 0.0f };
        // Wide enough to frame the debug geometry at the origin.
        camera.halfHeight = 2.0f;
        camera.zoomSpeed = 1.1f;
        return camera;
    }

    void OrthoCameraUpdate( OrthoCamera * camera, const OrthoCameraInput & input, i32 pixelHeight ) {
        if( input.zoomTicks != 0.0f ) {
            camera->halfHeight *= powf( camera->zoomSpeed, -input.zoomTicks );
            if( camera->halfHeight < kMinHalfHeight ) { camera->halfHeight = kMinHalfHeight; }
            if( camera->halfHeight > kMaxHalfHeight ) { camera->halfHeight = kMaxHalfHeight; }
        }

        if( input.panning && pixelHeight > 0 ) {
            Vec3 forward = {};
            Vec3 up = {};
            OrthoAxisBasis( camera->axis, &forward, &up );
            const Vec3 right = Vec3Normalize( Vec3Cross( forward, up ) );

            // The world is dragged, not the camera, so the centre moves against
            // the cursor. Screen y grows downward, which is why the up term is
            // added rather than subtracted.
            const f32 scale = ( 2.0f * camera->halfHeight ) / (f32)pixelHeight;
            camera->center = camera->center - right * ( input.panDeltaX * scale );
            camera->center = camera->center + up * ( input.panDeltaY * scale );
        }
    }

    Vec3 OrthoCameraScreenToWorld( const OrthoCamera & camera, f32 pixelX, f32 pixelY,
                                   i32 pixelWidth, i32 pixelHeight ) {
        Vec3 forward = {};
        Vec3 up = {};
        OrthoAxisBasis( camera.axis, &forward, &up );
        const Vec3 right = Vec3Normalize( Vec3Cross( forward, up ) );

        if( pixelHeight <= 0 ) {
            return camera.center;
        }

        // Same world-units-per-pixel the pan uses, so a drag and a click agree
        // about where the cursor is.
        const f32 scale = ( 2.0f * camera.halfHeight ) / (f32)pixelHeight;
        const f32 offsetX = pixelX - 0.5f * (f32)pixelWidth;
        const f32 offsetY = pixelY - 0.5f * (f32)pixelHeight;

        // Screen y grows downward and the up vector does not, hence the minus.
        Vec3 world = camera.center + right * ( offsetX * scale );
        world = world - up * ( offsetY * scale );
        return world;
    }

    Mat4 OrthoCameraViewProjection( const OrthoCamera & camera, i32 width, i32 height ) {
        Vec3 forward = {};
        Vec3 up = {};
        OrthoAxisBasis( camera.axis, &forward, &up );

        const Vec3 eye = camera.center - forward * kOrthoPullback;
        const Mat4 view = Mat4LookAt( eye, camera.center, up );

        const f32 aspect = height > 0 ? (f32)width / (f32)height : 1.0f;
        const Mat4 projection = Mat4Orthographic( camera.halfHeight * aspect, camera.halfHeight,
                                                  0.1f, 2.0f * kOrthoPullback );
        return projection * view;
    }

    Mat4 FlyCameraViewProjection( const FlyCamera & camera, i32 width, i32 height ) {
        const Vec3 forward = FlyCameraForward( camera );
        const Mat4 view = Mat4LookAt( camera.position, camera.position + forward,
                                      Vec3{ 0.0f, 1.0f, 0.0f } );
        const f32 aspect = height > 0 ? (f32)width / (f32)height : 1.0f;
        const Mat4 projection = Mat4Perspective( 60.0f * kDeg2Rad, aspect, 0.1f, 1000.0f );
        return projection * view;
    }

} // namespace sol
