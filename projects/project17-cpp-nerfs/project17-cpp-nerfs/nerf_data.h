#pragma once
#include "nerf_defines.h"

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
}
