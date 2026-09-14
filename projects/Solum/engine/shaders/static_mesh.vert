#version 450

layout( location = 0 ) in vec3 inPosition;
layout( location = 1 ) in vec3 inNormal;
layout( location = 2 ) in vec3 inColor;
layout( location = 3 ) in vec2 inUv;

layout( location = 0 ) out vec3 fragNormal;
layout( location = 1 ) out vec3 fragColor;
layout( location = 2 ) out vec2 fragUv;

// row_major because sol::Mat4 is row-major on the CPU and GLSL defaults to
// column-major. Without this the matrix arrives transposed.
//
// Must match the block in static_mesh.frag exactly: one range covers both
// stages, and the two declarations describe the same bytes.
layout( push_constant, row_major ) uniform PushConstants {
    mat4 mvp;
    vec4 tint;
} push;

void main() {
    gl_Position = push.mvp * vec4( inPosition, 1.0 );
    fragNormal = inNormal;
    fragColor = inColor;
    fragUv = inUv;
}
