#version 450

layout( location = 0 ) in vec3 fragNormal;
layout( location = 1 ) in vec3 fragColor;

layout( location = 0 ) out vec4 outColor;

void main() {
    // Normals ride through unused until there is a light to shade against.
    outColor = vec4( fragColor, 1.0 );
}
