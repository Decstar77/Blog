#include "nerf_data.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "stb_image.h"
#include "json.hpp"

using json = nlohmann::json;

namespace nerf {

    static void SetLargeString( LargeString * str, const std::string & value ) {
        i32 count = (i32)value.size();
        if( count > (i32)sizeof( str->data ) - 1 ) {
            count = (i32)sizeof( str->data ) - 1;
        }
        memcpy( str->data, value.data(), (size_t)count );
        str->data[count] = 0;
        str->count = count;
    }

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

    bool ReadNerfScene( const char * path, NerfScene * outScene ) {
        if( outScene == nullptr ) {
            return false;
        }
        *outScene = {};

        StringBuffer text = ReadEntireTextFile( path );
        if( text.data == nullptr ) {
            printf( "ReadNerfScene: could not open '%s'\n", path );
            return false;
        }

        json root = json::parse( text.data, nullptr, false );
        FreeStringBuffer( &text );

        if( root.is_discarded() || !root.is_object() ) {
            printf( "ReadNerfScene: '%s' is not valid json\n", path );
            return false;
        }

        if( !root.contains( "camera_angle_x" ) || !root.contains( "frames" ) ) {
            printf( "ReadNerfScene: '%s' is missing camera_angle_x or frames\n", path );
            return false;
        }

        const json & frames = root[ "frames" ];
        if( !frames.is_array() ) {
            printf( "ReadNerfScene: '%s' has a non-array frames field\n", path );
            return false;
        }

        NerfScene scene = {};
        scene.cameraAngleX = root[ "camera_angle_x" ].get<f32>();
        scene.frameCount = (i32)frames.size();
        scene.frames = (NerfFrame *)calloc( (size_t)scene.frameCount, sizeof( NerfFrame ) );
        if( scene.frames == nullptr && scene.frameCount > 0 ) {
            return false;
        }

        for( i32 i = 0; i < scene.frameCount; i++ ) {
            const json & src = frames[ (size_t)i ];
            NerfFrame & dst = scene.frames[i];

            dst.transformMatrix = Mat4Identity();

            if( src.contains( "file_path" ) ) {
                SetLargeString( &dst.filePath, src[ "file_path" ].get<std::string>() );
            }
            if( src.contains( "rotation" ) ) {
                dst.rotation = src[ "rotation" ].get<f32>();
            }

            if( !src.contains( "transform_matrix" ) ) {
                printf( "ReadNerfScene: frame %d has no transform_matrix\n", i );
                FreeNerfScene( &scene );
                return false;
            }

            const json & matrix = src[ "transform_matrix" ];
            if( !matrix.is_array() || matrix.size() != 4 ) {
                printf( "ReadNerfScene: frame %d has a malformed transform_matrix\n", i );
                FreeNerfScene( &scene );
                return false;
            }

            for( i32 row = 0; row < 4; row++ ) {
                const json & values = matrix[ (size_t)row ];
                if( !values.is_array() || values.size() != 4 ) {
                    printf( "ReadNerfScene: frame %d row %d has a malformed transform_matrix\n", i, row );
                    FreeNerfScene( &scene );
                    return false;
                }
                for( i32 col = 0; col < 4; col++ ) {
                    dst.transformMatrix.m[row][col] = values[ (size_t)col ].get<f32>();
                }
            }
        }

        *outScene = scene;
        return true;
    }

    void FreeNerfScene( NerfScene * scene ) {
        if( scene == nullptr ) {
            return;
        }
        free( scene->frames );
        *scene = {};
    }
}
