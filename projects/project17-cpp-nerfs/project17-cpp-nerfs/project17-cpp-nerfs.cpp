#include "nerf_data.h"
#include "nerf_math.h"
#include "nerf_mlp.h"

#include <cstdio>
#include <iostream>

using namespace nerf;

static void Test001_LoadingScenes( const char * path ) {
    nerf::NerfScene scene = {};
    if ( !nerf::ReadNerfScene( path, &scene ) ) {
        return;
    }

    printf( "%s\n", path );
    printf( "camera_angle_x = %f rad (%f deg), %d frames\n", scene.cameraAngleX, scene.cameraAngleX * nerf::kRad2Deg, scene.frameCount );
    printf( "focal length at 800px wide = %f\n", nerf::FocalLengthFromFovX( scene.cameraAngleX, 800 ) );

    // Cameras in this dataset sit on a sphere looking at the origin, so the
    // distance to the origin is a cheap sanity check that the matrices parsed.
    nerf::f32 minRadius = 1e30f;
    nerf::f32 maxRadius = 0.0f;
    for ( nerf::i32 i = 0; i < scene.frameCount; i++ ) {
        const nerf::Vec3 eye = nerf::Mat4Translation( scene.frames[i].transformMatrix );
        const nerf::f32 radius = nerf::Vec3Length( eye );
        minRadius = radius < minRadius ? radius : minRadius;
        maxRadius = radius > maxRadius ? radius : maxRadius;
    }

    printf( "camera distance from origin: min %f, max %f\n", minRadius, maxRadius );
    for ( nerf::i32 i = 0; i < scene.frameCount && i < 3; i++ ) {
        const nerf::NerfFrame & frame = scene.frames[i];
        const nerf::Vec3 eye = nerf::Mat4Translation( frame.transformMatrix );
        const nerf::Vec3 fwd = nerf::Mat4Forward( frame.transformMatrix );
        printf( "  [%d] %-20s rotation %f eye ( %f, %f, %f ) forward ( %f, %f, %f )\n", i, frame.filePath.data, frame.rotation, eye.x, eye.y, eye.z, fwd.x, fwd.y, fwd.z );
    }

    nerf::FreeNerfScene( &scene );
}

static void Test002_ImageRegression() {
    const i32 layers = 3;
    const i32 mlpsizes[layers] = { 2, 1000, 4 };
    NetworkMlp * mlp = MlpCreate( mlpsizes, layers, ACTIVATION_FUNCTION_TANH, ACTIVATION_FUNCTION_SIGMOID );
    
    Image image = ReadEntireImage( "C:/Projects/2025/Blog/data/nerf_synthetic/lego/test/r_0.png" );

    for ( i32 y = 0; y < image.height; y++ ) {
        for ( i32 x = 0; x < image.width; x++ ) {
            const f32 v = f32( y ) / f32( image.height );
            const f32 u = f32( x ) / f32( image.width );
            const Vec4 c = Fetch( &image, x, y );

            f32 in[2] = { u, v };
            f32 out[4] = { 0, 0, 0, 0 };

            MlpZeroGrads( mlp );
            MlpForward( mlp, in, out );
            f32 loss = Mlp_LossMSE( mlp, &c.x, out );
            MlpBackward( mlp );
            MlpApplyGrads( mlp, 0.001f );
        }
    }

    FreeImage( &image );
}

int main( int argc, char ** argv ) {
    const char * path = argc > 1 ? argv[1] : "C:/Projects/2025/Blog/data/nerf_synthetic/lego/transforms_train.json";



    system( "PAUSE" );
    return 0;
}
