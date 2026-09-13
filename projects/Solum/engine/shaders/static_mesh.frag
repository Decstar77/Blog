#version 450

layout( location = 0 ) in vec3 fragNormal;
layout( location = 1 ) in vec3 fragColor;
layout( location = 2 ) in vec2 fragUv;

layout( location = 0 ) out vec4 outColor;

// Set 0 binding 0: every mesh binds one, even untextured meshes, which bind
// the renderer's white 1x1 fallback so this sample is always valid.
layout( set = 0, binding = 0 ) uniform sampler2D texSampler;

void main() {
    // Normals ride through unused until there is a light to shade against.
    vec4 texel = texture( texSampler, fragUv );
    outColor = vec4( fragColor, 1.0 ) * texel;
}
