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

// Maps a pixel centre to the [ -1, 1 ] range the encoding expects, then runs the frequency bands
// over it. encoded has to be EncodingOutputCount( enc ) wide.
static void EncodePixel( const PositionalEncoding & enc, i32 x, i32 y, i32 width, i32 height, f32 * encoded ) {
    const f32 u = ( f32( x ) + 0.5f ) / f32( width ) * 2.0f - 1.0f;
    const f32 v = ( f32( y ) + 0.5f ) / f32( height ) * 2.0f - 1.0f;
    const f32 uv[2] = { u, v };
    EncodingApply( enc, uv, encoded );
}

static void Test002_ImageRegression( const char * path ) {
    // 10 bands over a 2d input is 2 + 2 * 2 * 10 = 42 inputs, the same count the NeRF paper uses
    // for positions.
    const PositionalEncoding enc = EncodingCreate( 2, 10, true );
    const i32 encodedCount = EncodingOutputCount( enc );

    const i32 layers = 4;
    const i32 mlpsizes[layers] = { encodedCount, 128, 128, 4 };

    Image image = ReadEntireImage( path );
    NetworkMlp * mlp = MlpCreate( mlpsizes, layers, ACTIVATION_FUNCTION_RELU, ACTIVATION_FUNCTION_SIGMOID );

    MlpSetOptimizerAdam( mlp );

    const i32 iterations = 2000;
    const i32 batchSize = 256;
    const f32 lr = 0.01f;

    RandomSeries rng = RandomSeed( 0x1234u );
    for ( i32 iter = 0; iter < iterations; iter++ ) {
        f32 batchLoss = 0.0f;

        for ( i32 s = 0; s < batchSize; s++ ) {
            const i32 x = RandomBelow( &rng, image.width );
            const i32 y = RandomBelow( &rng, image.height );
            const Vec4 c = Fetch( &image, x, y );

            f32 in[kMlpMaxWidth] = {};
            EncodePixel( enc, x, y, image.width, image.height, in );

            f32 out[4] = { 0, 0, 0, 0 };
            f32 dLdOut[4] = { 0, 0, 0, 0 };

            MlpForward( mlp, in, out );
            batchLoss += MlpLossMSE( mlp, &c.x, dLdOut );
            MlpBackward( mlp, dLdOut );
        }

        MlpApplyGrads( mlp, lr );

        if ( iter % 200 == 0 || iter == iterations - 1 ) {
            printf( "  iter %4d  batch loss %f\n", iter, batchLoss / f32( batchSize ) );
        }
    }

    Image predicted = {};
    predicted.width = image.width;
    predicted.height = image.height;
    predicted.channels = 4;
    predicted.pixels = (f32 *)malloc( (size_t)predicted.width * predicted.height * predicted.channels * sizeof( f32 ) );

    f64 total = 0.0;
    for ( i32 y = 0; y < image.height; y++ ) {
        for ( i32 x = 0; x < image.width; x++ ) {
            const Vec4 c = Fetch( &image, x, y );

            f32 in[kMlpMaxWidth] = {};
            EncodePixel( enc, x, y, image.width, image.height, in );

            f32 out[4] = { 0, 0, 0, 0 };
            MlpForward( mlp, in, out );
            total += MlpLossMSE( mlp, &c.x, nullptr );

            if ( predicted.pixels != nullptr ) {
                f32 * dst = predicted.pixels + ( (size_t)y * predicted.width + x ) * predicted.channels;
                dst[0] = out[0];
                dst[1] = out[1];
                dst[2] = out[2];
                dst[3] = out[3];
            }
        }
    }

    const f64 mse = total / f64( image.width * image.height );
    printf( "  full image mse %f, psnr %f dB\n", mse, 10.0 * log10( 1.0 / ( mse > 1e-12 ? mse : 1e-12 ) ) );

    if ( predicted.pixels != nullptr ) {
        const char * outPath = "test002_predicted.png";
        if ( WriteImagePng( outPath, &predicted ) ) {
            printf( "  wrote %s ( %dx%d )\n", outPath, predicted.width, predicted.height );
        }
        free( predicted.pixels );
    }

    MlpDestroy( mlp );
    FreeImage( &image );
}

int main( int argc, char ** argv ) {
    const char * path = argc > 1 ? argv[1] : "C:/Projects/2025/Blog/data/nerf_synthetic/lego/transforms_train.json";

    Test002_ImageRegression( "C:/Projects/2025/Blog/data/nerf_synthetic/lego/test/r_0.png" );

    system( "PAUSE" );
    return 0;
}
