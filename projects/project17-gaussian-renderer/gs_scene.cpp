#include "gs_scene.h"

#include "gs_list.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdio>

// A fixed demo scene, built to exercise the parts of the renderer a single splat cannot: thousands of
// splats spread over most of the tile grid, translucent shells that only look right when the per-tile
// depth sort is correct, and anisotropic splats whose rotation has to survive the covariance path.
// Nothing here is random per run, so two frames of the same camera are comparable.

constexpr u32 kSeed = 0x9E3779B9u;

constexpr int   kFloorSide = 48;      // splats per side of the chequerboard
constexpr float kFloorExtent = 7.0f;  // half-width, world units
constexpr float kFloorHeight = -1.3f;

constexpr int kShellSplats = 2800;    // per sphere
constexpr int kRingSplats = 900;

struct Rng {
    u32 state;
};

/*
===================
===================
*/
static float rng_unit( Rng * rng ) {
    // xorshift32. This only drives jitter, so the quality bar is "not visibly periodic".
    u32 x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return float( x >> 8 ) * ( 1.0f / 16777216.0f );
}

/*
===================
===================
*/
static float rng_range( Rng * rng, float lo, float hi ) {
    return lo + ( hi - lo ) * rng_unit( rng );
}

/*
===================
===================
*/
static glm::mat3 basis_from_normal( glm::vec3 n ) {
    const glm::vec3 guide = fabsf( n.y ) < 0.99f ? glm::vec3( 0, 1, 0 ) : glm::vec3( 1, 0, 0 );
    const glm::vec3 tangent = glm::normalize( glm::cross( guide, n ) );
    const glm::vec3 bitangent = glm::cross( n, tangent );
    return glm::mat3( tangent, bitangent, n );
}

/*
===================
===================
*/
static void add_splat( Scene * scene, glm::vec3 position, glm::mat3 rotation, glm::vec3 scale, glm::vec4 colour ) {
    Gaussian g;
    g.position = position;
    g.rotation = rotation;
    g.scale = scale;
    g.colour = colour;
    list_add( scene->gaussians, g );
}

/*
===================
===================
*/
static void build_floor( Scene * scene, Rng * rng ) {
    const glm::mat3 rotation = basis_from_normal( glm::vec3( 0, 1, 0 ) );
    const float step = 2.0f * kFloorExtent / float( kFloorSide - 1 );

    for ( int iz = 0; iz < kFloorSide; iz++ ) {
        for ( int ix = 0; ix < kFloorSide; ix++ ) {
            const float x = -kFloorExtent + step * float( ix );
            const float z = -kFloorExtent + step * float( iz );

            // Fade the board out rather than ending it on a hard square edge.
            const float distance = sqrtf( x * x + z * z );
            const float fade = 1.0f - Min( 1.0f, distance / kFloorExtent );
            if ( fade <= 0.02f ) {
                continue;
            }

            const float tone = ( ( ix ^ iz ) & 1 ) ? 0.34f : 0.16f;
            const glm::vec3 position( x + rng_range( rng, -0.02f, 0.02f ), kFloorHeight, z + rng_range( rng, -0.02f, 0.02f ) );
            const glm::vec3 scale( 0.5f * step, 0.5f * step, 0.008f );
            const glm::vec4 colour( tone, tone * 1.02f, tone * 1.15f, 0.9f * fade * fade );

            add_splat( scene, position, rotation, scale, colour );
        }
    }
}

/*
===================
===================
*/
static void build_shell( Scene * scene, Rng * rng, glm::vec3 centre, float radius, glm::vec3 tint ) {
    const glm::vec3 light = glm::normalize( glm::vec3( 0.4f, 0.85f, 0.55f ) );

    // Fibonacci sphere: even coverage without the pole clustering a lat/long loop gives.
    const float golden = kPi * ( 3.0f - sqrtf( 5.0f ) );

    for ( int i = 0; i < kShellSplats; i++ ) {
        const float y = 1.0f - 2.0f * ( float( i ) + 0.5f ) / float( kShellSplats );
        const float ring = sqrtf( Max( 0.0f, 1.0f - y * y ) );
        const float theta = golden * float( i );

        const glm::vec3 normal( cosf( theta ) * ring, y, sinf( theta ) * ring );
        const glm::vec3 position = centre + normal * ( radius + rng_range( rng, -0.015f, 0.015f ) );

        const float lambert = Max( 0.0f, glm::dot( normal, light ) );
        const float shade = 0.40f + 0.60f * lambert;

        const glm::vec3 scale( 0.055f * radius / 0.95f, 0.055f * radius / 0.95f, 0.010f );
        const glm::vec4 colour( tint.x * shade, tint.y * shade, tint.z * shade, 0.30f );

        add_splat( scene, position, basis_from_normal( normal ), scale, colour );
    }
}

/*
===================
===================
*/
static void build_ring( Scene * scene, Rng * rng, glm::vec3 centre, float radius ) {
    const glm::mat3 tilt = glm::mat3( glm::rotate( glm::mat4( 1.0f ), 0.42f, glm::normalize( glm::vec3( 1.0f, 0.0f, 0.35f ) ) ) );

    for ( int i = 0; i < kRingSplats; i++ ) {
        const float t = kTwoPi * float( i ) / float( kRingSplats );
        const float r = radius + rng_range( rng, -0.05f, 0.05f );

        const glm::vec3 radial( cosf( t ), 0.0f, sinf( t ) );
        const glm::vec3 tangent( -sinf( t ), 0.0f, cosf( t ) );
        const glm::vec3 up( 0.0f, 1.0f, 0.0f );

        const glm::vec3 position = centre + tilt * ( radial * r ) + glm::vec3( 0.0f, rng_range( rng, -0.03f, 0.03f ), 0.0f );
        const glm::mat3 rotation( tilt * tangent, tilt * up, tilt * radial );

        // Long along the tangent, thin across it: the ring reads as a band rather than a string of beads.
        const glm::vec3 scale( 0.075f, 0.014f, 0.030f );

        const float warmth = 0.5f + 0.5f * cosf( t * 3.0f );
        const glm::vec4 colour( 1.0f, 0.72f + 0.20f * warmth, 0.30f + 0.25f * warmth, 0.85f );

        add_splat( scene, position, rotation, scale, colour );
    }
}

/*
===================
===================
*/
void scene_frame_camera( Scene * scene ) {
    const i32 count = scene->gaussians.count;
    if ( count <= 0 ) {
        return;
    }

    glm::dvec3 sum( 0.0 );
    for ( i32 i = 0; i < count; i++ ) {
        sum += glm::dvec3( scene->gaussians[i].position );
    }
    const glm::vec3 centre( sum / double( count ) );

    // A bounding box would be dragged out to the horizon by the stray splats a capture leaves behind,
    // so size the framing by spread instead.
    glm::dvec3 variance( 0.0 );
    for ( i32 i = 0; i < count; i++ ) {
        const glm::dvec3 d = glm::dvec3( scene->gaussians[i].position ) - glm::dvec3( centre );
        variance += d * d;
    }
    variance /= double( count );

    const float spread = sqrtf( Max( Max( float( variance.x ), float( variance.y ) ), float( variance.z ) ) );
    const float distance = Max( 0.5f, 2.5f * spread );

    scene->camera.position = centre + glm::vec3( 0.0f, 0.35f * spread, distance );
    scene->camera.yaw = 0.0f;
    scene->camera.pitch = 0.0f;
    camera_refresh( &scene->camera );

    printf( "scene: %d splats, centre (%.2f, %.2f, %.2f), spread %.2f\n", count, centre.x, centre.y, centre.z, spread );
}

/*
===================
===================
*/
void scene_build_demo( Scene * scene ) {
    Rng rng = { kSeed };

    list_reserve( scene->gaussians, kFloorSide * kFloorSide + 3 * kShellSplats + kRingSplats );

    build_floor( scene, &rng );

    // Spread in depth as well as across the screen, and overlapping from the default camera, so the
    // shells interleave instead of sorting into three tidy layers.
    build_shell( scene, &rng, glm::vec3( -1.85f, 0.15f, 0.55f ), 0.95f, glm::vec3( 1.00f, 0.38f, 0.28f ) );
    build_shell( scene, &rng, glm::vec3( 0.20f, 0.40f, -1.15f ), 1.15f, glm::vec3( 0.22f, 0.78f, 0.82f ) );
    build_shell( scene, &rng, glm::vec3( 1.95f, -0.05f, 0.80f ), 0.85f, glm::vec3( 0.64f, 0.42f, 0.96f ) );

    build_ring( scene, &rng, glm::vec3( 0.20f, 0.40f, -1.15f ), 1.85f );

    scene->camera.position = glm::vec3( 0.0f, 0.9f, 6.4f );
    scene->camera.yaw = 0.0f;
    scene->camera.pitch = -0.10f;
    camera_refresh( &scene->camera );

    scene->gaussians_dirty = true;
}
