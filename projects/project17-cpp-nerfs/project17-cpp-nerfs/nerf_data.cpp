#include "nerf_data.h"

#include <cstdio>
#include <cstdlib>

#include "stb_image.h"

namespace nerf {

    StringBuffer ReadEntireTextFile( const char * path ) {
        StringBuffer buffer = {};

        FILE * file = nullptr;
        if( fopen_s( &file, path, "rb" ) != 0 || file == nullptr ) {
            return buffer;
        }

        fseek( file, 0, SEEK_END );
        const long size = ftell( file );
        fseek( file, 0, SEEK_SET );

        if( size < 0 ) {
            fclose( file );
            return buffer;
        }

        buffer.cap = (i32)size + 1;
        buffer.data = (char *)malloc( (size_t)buffer.cap );
        if( buffer.data == nullptr ) {
            buffer.cap = 0;
            fclose( file );
            return buffer;
        }

        buffer.count = (i32)fread( buffer.data, 1, (size_t)size, file );
        buffer.data[ buffer.count ] = 0;
        fclose( file );

        return buffer;
    }

    void FreeStringBuffer( StringBuffer * buffer ) {
        if( buffer == nullptr ) {
            return;
        }
        free( buffer->data );
        *buffer = {};
    }

    Image ReadEntireImage( const char * path ) {
        Image image = {};
        image.pixels = stbi_loadf( path, &image.width, &image.height, &image.channels, 0 );
        if( image.pixels == nullptr ) {
            image = {};
        }
        return image;
    }

    void FreeImage( Image * image ) {
        if( image == nullptr ) {
            return;
        }
        stbi_image_free( image->pixels );
        *image = {};
    }
}
