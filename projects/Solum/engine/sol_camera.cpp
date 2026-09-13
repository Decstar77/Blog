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

    Mat4 FlyCameraViewProjection( const FlyCamera & camera, i32 width, i32 height ) {
        const Vec3 forward = FlyCameraForward( camera );
        const Mat4 view = Mat4LookAt( camera.position, camera.position + forward,
                                      Vec3{ 0.0f, 1.0f, 0.0f } );
        const f32 aspect = height > 0 ? (f32)width / (f32)height : 1.0f;
        const Mat4 projection = Mat4Perspective( 60.0f * kDeg2Rad, aspect, 0.1f, 1000.0f );
        return projection * view;
    }

} // namespace sol
