#version 450

layout( location = 0 ) in vec3 fragNormal;
layout( location = 1 ) in vec3 fragColor;
layout( location = 2 ) in vec2 fragUv;

layout( location = 0 ) out vec4 outColor;

// Set 0 binding 0: every mesh binds one, even untextured meshes, which bind
// the renderer's white 1x1 fallback so this sample is always valid.
layout( set = 0, binding = 0 ) uniform sampler2D texSampler;

// Must match the block in static_mesh.vert exactly - see the note there.
layout( push_constant, row_major ) uniform PushConstants {
    mat4 mvp;
    vec4 tint;
} push;

// A fixed key light, until there are real lights in the world. Without it a
// solid-coloured box renders as one flat silhouette with no visible edges,
// which makes modelling in the viewport impossible to judge.
const vec3 kLightDirection = normalize( vec3( 0.4, 0.8, 0.35 ) );
const float kAmbient = 0.35;

void main() {
    vec4 texel = texture( texSampler, fragUv );

    // Faces pointing away from the light keep the ambient term rather than
    // going black, so geometry in shadow is still readable.
    vec3 normal = normalize( fragNormal );
    float diffuse = max( dot( normal, kLightDirection ), 0.0 );
    float lighting = kAmbient + ( 1.0 - kAmbient ) * diffuse;

    // The tint is a multiply, so a selected object brightens and shifts without
    // losing the shading that says which way its faces point.
    outColor = vec4( fragColor * lighting, 1.0 ) * texel * push.tint;
}
