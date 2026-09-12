#version 450

layout( location = 0 ) in vec3 inPosition;
layout( location = 1 ) in vec3 inNormal;
layout( location = 2 ) in vec3 inColor;

layout( location = 0 ) out vec3 fragNormal;
layout( location = 1 ) out vec3 fragColor;

// row_major because sol::Mat4 is row-major on the CPU and GLSL defaults to
// column-major. Without this the matrix arrives transposed.
layout( push_constant, row_major ) uniform PushConstants {
    mat4 mvp;
} push;

void main() {
    gl_Position = push.mvp * vec4( inPosition, 1.0 );
    fragNormal = inNormal;
    fragColor = inColor;
}
