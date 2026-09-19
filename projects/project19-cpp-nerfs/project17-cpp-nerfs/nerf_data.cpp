#include "nerf_data.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "stb_image.h"
#include "stb_image_write.h"
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

    bool WriteImagePng( const char * path, const Image * image ) {
        if( image == nullptr || image->pixels == nullptr || image->channels <= 0 ) {
            return false;
        }

        const i32 count = image->width * image->height * image->channels;
        u8 * bytes = (u8 *)malloc( (size_t)count );
        if( bytes == nullptr ) {
            return false;
        }

        for( i32 i = 0; i < count; i++ ) {
            f32 v = image->pixels[i];
            v = v < 0.0f ? 0.0f : ( v > 1.0f ? 1.0f : v );
            // stbi_loadf raises 8-bit pngs to gamma 2.2 on load, so go back the other way here.
            // Alpha is stored linearly by stb, so it is written straight through.
            const bool isAlpha = ( image->channels == 4 || image->channels == 2 ) && ( i % image->channels ) == image->channels - 1;
            const f32 encoded = isAlpha ? v : powf( v, 1.0f / 2.2f );
            bytes[i] = (u8)( encoded * 255.0f + 0.5f );
        }

        const i32 ok = stbi_write_png( path, image->width, image->height, image->channels, bytes, image->width * image->channels );
        free( bytes );

        if( ok == 0 ) {
            printf( "WriteImagePng: could not write '%s'\n", path );
            return false;
        }
        return true;
    }

    Vec4 Fetch( Image * image, i32 x, i32 y ) {
        i32 idx = ( y * image->width + x ) * image->channels;
        f32 a = 1;
        if ( image->channels == 4 ) {
            a = image->pixels[idx + 3];
        }
        return Vec4 { image->pixels[idx], image->pixels[idx + 1], image->pixels[idx + 2], a };
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
