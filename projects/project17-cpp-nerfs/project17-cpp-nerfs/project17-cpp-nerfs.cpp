#include "nerf_data.h"
#include "nerf_math.h"

#include <cstdio>
#include <iostream>

int main( int argc, char ** argv ) {
    const char * path = argc > 1 ? argv[1] : "C:/Projects/2025/Blog/data/nerf_synthetic/lego/transforms_train.json";

    nerf::NerfScene scene = {};
    if( !nerf::ReadNerfScene( path, &scene ) ) {
        return 1;
    }

    printf( "%s\n", path );
    printf( "camera_angle_x = %f rad (%f deg), %d frames\n", scene.cameraAngleX, scene.cameraAngleX * nerf::kRad2Deg, scene.frameCount );
    printf( "focal length at 800px wide = %f\n", nerf::FocalLengthFromFovX( scene.cameraAngleX, 800 ) );

    // Cameras in this dataset sit on a sphere looking at the origin, so the
    // distance to the origin is a cheap sanity check that the matrices parsed.
    nerf::f32 minRadius = 1e30f;
    nerf::f32 maxRadius = 0.0f;
    for( nerf::i32 i = 0; i < scene.frameCount; i++ ) {
        const nerf::Vec3 eye = nerf::Mat4Translation( scene.frames[i].transformMatrix );
        const nerf::f32 radius = nerf::Vec3Length( eye );
        minRadius = radius < minRadius ? radius : minRadius;
        maxRadius = radius > maxRadius ? radius : maxRadius;
    }

    printf( "camera distance from origin: min %f, max %f\n", minRadius, maxRadius );
    for( nerf::i32 i = 0; i < scene.frameCount && i < 3; i++ ) {
        const nerf::NerfFrame & frame = scene.frames[i];
        const nerf::Vec3 eye = nerf::Mat4Translation( frame.transformMatrix );
        const nerf::Vec3 fwd = nerf::Mat4Forward( frame.transformMatrix );
        printf( "  [%d] %-20s rotation %f eye ( %f, %f, %f ) forward ( %f, %f, %f )\n", i, frame.filePath.data, frame.rotation, eye.x, eye.y, eye.z, fwd.x, fwd.y, fwd.z );
    }

    nerf::FreeNerfScene( &scene );
    system( "PAUSE" );
    return 0;
}
