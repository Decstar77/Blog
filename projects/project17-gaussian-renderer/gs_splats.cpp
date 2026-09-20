#include "gs_splats.h"
#include "gs_window.h"

#include <glm/ext/matrix_clip_space.hpp>

#include <cmath>

constexpr float kLookSensitivity = 0.0025f;  // radians per pixel of mouse travel
constexpr float kMoveSpeed = 2.0f;           // world units per second
constexpr float kFastMultiplier = 4.0f;
constexpr float kPitchLimit = kHalfPi - 0.01f;

/*
===================
===================
*/
void camera_refresh( Camera * camera ) {
    const float cy = cosf( camera->yaw );
    const float sy = sinf( camera->yaw );
    const float cp = cosf( camera->pitch );
    const float sp = sinf( camera->pitch );

    const glm::vec3 forward( -sy * cp, sp, -cy * cp );
    const glm::vec3 right( cy, 0.0f, -sy );
    const glm::vec3 up = glm::cross( right, forward );
    camera->rotation = glm::mat3( right, up, -forward );
}

/*
===================
===================
*/
void camera_update( Camera * camera, const GsInput & input, float dt ) {
    camera->yaw -= input.mouse_dx * kLookSensitivity;
    camera->pitch -= input.mouse_dy * kLookSensitivity;
    camera->pitch = Min( Max( camera->pitch, -kPitchLimit ), kPitchLimit );

    if ( camera->yaw > kPi ) {
        camera->yaw -= kTwoPi;
    } else if ( camera->yaw < -kPi ) {
        camera->yaw += kTwoPi;
    }

    camera_refresh( camera );

    const glm::vec3 right = camera->rotation[0];
    const glm::vec3 up = camera->rotation[1];
    const glm::vec3 forward = -camera->rotation[2];

    glm::vec3 move = right * input.move_right + up * input.move_up + forward * input.move_forward;
    const float length = glm::length( move );
    if ( length <= 0.0f ) {
        return;
    }

    move /= length;
    camera->position += move * ( kMoveSpeed * ( input.fast ? kFastMultiplier : 1.0f ) * dt );
}
