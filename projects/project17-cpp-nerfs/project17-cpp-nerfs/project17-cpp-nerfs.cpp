#include "nerf_data.h"
#include "nerf_math.h"
#include "nerf_mlp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// Fit a tiny MLP to a single image: (u, v) -> (r, g, b, a). This is the sanity check that the
// forward pass, the loss and the backward pass agree, and it is also the motivating failure case
// for NeRF's positional encoding: raw coordinates only let the net reach the low frequencies, so
// the loss flattens out well short of zero and the fit stays blurry.
static void Test002_ImageRegression( const char * path ) {
    Image image = ReadEntireImage( path );
    if ( image.pixels == nullptr ) {
        printf( "Test002: could not open '%s'\n", path );
        return;
    }

    const i32 layers = 3;
    const i32 mlpsizes[layers] = { 2, 256, 4 };
    NetworkMlp * mlp = MlpCreate( mlpsizes, layers, ACTIVATION_FUNCTION_TANH, ACTIVATION_FUNCTION_SIGMOID );
    if ( mlp == nullptr ) {
        printf( "Test002: MlpCreate failed\n" );
        FreeImage( &image );
        return;
    }
    MlpSetOptimizerAdam( mlp );

    printf( "%s: %dx%d, %d channels\n", path, image.width, image.height, image.channels );

    const i32 iterations = 2000;
    const i32 batchSize = 256;
    const f32 lr = 0.01f;

    RandomSeries rng = RandomSeed( 0x1234u );
    for ( i32 iter = 0; iter < iterations; iter++ ) {
        f32 batchLoss = 0.0f;

        // One MlpZeroGrads is not needed here: MlpApplyGrads clears the buffers after every step,
        // and MlpCreate starts them at zero.
        for ( i32 s = 0; s < batchSize; s++ ) {
            const i32 x = RandomBelow( &rng, image.width );
            const i32 y = RandomBelow( &rng, image.height );
            const f32 u = ( f32( x ) + 0.5f ) / f32( image.width );
            const f32 v = ( f32( y ) + 0.5f ) / f32( image.height );
            const Vec4 c = Fetch( &image, x, y );

            const f32 in[2] = { u, v };
            f32 out[4] = { 0, 0, 0, 0 };
            f32 dLdOut[4] = { 0, 0, 0, 0 };

            MlpForward( mlp, in, out );
            batchLoss += MlpLossMSE( mlp, &c.x, dLdOut );
            MlpBackward( mlp, dLdOut );
        }

        // The gradients summed over the batch, so step once here; MlpApplyGrads does the averaging.
        MlpApplyGrads( mlp, lr );

        if ( iter % 200 == 0 || iter == iterations - 1 ) {
            printf( "  iter %4d  batch loss %f\n", iter, batchLoss / f32( batchSize ) );
        }
    }

    // Final pass over every pixel, so the number is not just whatever the last random batch was.
    f64 total = 0.0;
    for ( i32 y = 0; y < image.height; y++ ) {
        for ( i32 x = 0; x < image.width; x++ ) {
            const f32 u = ( f32( x ) + 0.5f ) / f32( image.width );
            const f32 v = ( f32( y ) + 0.5f ) / f32( image.height );
            const Vec4 c = Fetch( &image, x, y );

            const f32 in[2] = { u, v };
            MlpForward( mlp, in, nullptr );
            total += MlpLossMSE( mlp, &c.x, nullptr );
        }
    }

    const f64 mse = total / f64( image.width * image.height );
    printf( "  full image mse %f, psnr %f dB\n", mse, 10.0 * log10( 1.0 / ( mse > 1e-12 ? mse : 1e-12 ) ) );

    MlpDestroy( mlp );
    FreeImage( &image );
}

int main( int argc, char ** argv ) {
    const char * path = argc > 1 ? argv[1] : "C:/Projects/2025/Blog/data/nerf_synthetic/lego/transforms_train.json";

    Test001_LoadingScenes( path );
    printf( "\n" );
    Test002_ImageRegression( "C:/Projects/2025/Blog/data/nerf_synthetic/lego/test/r_0.png" );

    system( "PAUSE" );
    return 0;
}
