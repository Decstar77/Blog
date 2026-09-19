#pragma once

#include <glm/glm.hpp>
#include "gs_list.h"
#include "gs_window.h"

struct Gaussian {
    glm::vec3 position;
    glm::vec3 scale;
    glm::mat3 rotation;
    glm::vec4 colour;
};

struct Camera {
    glm::vec3 position;
    glm::mat3 rotation;  // camera -> world basis: columns are right, up, back
    float     yaw;       // radians, around world +Y, 0 looks down -Z
    float     pitch;     // radians, clamped just shy of straight up/down
};

void camera_refresh( Camera * camera );
void camera_update( Camera * camera, const GsInput & input, float dt );

struct Scene {
    Camera          camera;
    List<Gaussian>  gaussians;
};

