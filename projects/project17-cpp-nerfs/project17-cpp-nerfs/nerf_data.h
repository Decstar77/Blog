#pragma once
#include "nerf_defines.h"
#include "nerf_math.h"

namespace nerf {
    struct Image {
        i32     width;
        i32     height;
        i32     channels;
        f32 *   pixels;
    };

    StringBuffer    ReadEntireTextFile( const char * path );
    void            FreeStringBuffer( StringBuffer * buffer );

    Image           ReadEntireImage( const char * path );
    void            FreeImage( Image * image );

    Vec4            Fetch( Image * image, i32 x, i32 y );

    // One entry of the "frames" array in a NeRF-synthetic transforms_*.json.
    struct NerfFrame {
        LargeString filePath;           // relative to the transforms file, e.g. "./train/r_0" (no extension)
        f32         rotation;           // radians, unused by the renderer but present in the dataset
        Mat4        transformMatrix;    // camera-to-world, OpenGL convention
    };

    // A whole transforms_train.json / transforms_val.json / transforms_test.json.
    struct NerfScene {
        f32         cameraAngleX;       // horizontal field of view, radians
        NerfFrame * frames;
        i32         frameCount;
    };

    bool            ReadNerfScene( const char * path, NerfScene * outScene );
    void            FreeNerfScene( NerfScene * scene );
}
