#pragma once

#include <glm/glm.hpp>
#include "gs_list.h"

struct Gaussian {
    glm::vec3 position;
    glm::vec3 scale;
    glm::mat3 rotation;
    glm::vec4 colour;
};

struct Camera {
    glm::vec3 position;
    glm::mat3 rotation;
};

struct Scene {
    Camera          camera;
    List<Gaussian>  gaussians;
};

