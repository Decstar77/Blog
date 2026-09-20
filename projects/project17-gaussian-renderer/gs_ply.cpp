#include "gs_ply.h"

#include "gs_list.h"

#include <glm/gtc/quaternion.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Nothing in the file is stored in the form the renderer wants: scales are logs, opacity is a logit,
// the colour is the degree-0 spherical harmonic coefficient, and the rotation is an unnormalised
// quaternion. Exporters also disagree on property order and on whether the vestigial normals are
// present, so the layout is taken from the header rather than assumed.

constexpr float kShC0 = 0.28209479177387814f;  // the degree-0 SH basis function, a constant over the sphere
constexpr int   kMaxProperties = 128;
constexpr int   kChunkVertices = 8192;         // keeps the staging buffer small next to an 800 MB file

enum PlyScalar {
    kPlyUnknown = 0,
    kPlyI8,
    kPlyU8,
    kPlyI16,
    kPlyU16,
    kPlyI32,
    kPlyU32,
    kPlyF32,
    kPlyF64,
};

struct PlyProperty {
    char      name[32];
    PlyScalar scalar;
    int       offset;
};

/*
===================
===================
*/
static PlyScalar ply_scalar_from_name( const char * name ) {
    if ( strcmp( name, "float" ) == 0 || strcmp( name, "float32" ) == 0 ) return kPlyF32;
    if ( strcmp( name, "double" ) == 0 || strcmp( name, "float64" ) == 0 ) return kPlyF64;
    if ( strcmp( name, "char" ) == 0 || strcmp( name, "int8" ) == 0 ) return kPlyI8;
    if ( strcmp( name, "uchar" ) == 0 || strcmp( name, "uint8" ) == 0 ) return kPlyU8;
    if ( strcmp( name, "short" ) == 0 || strcmp( name, "int16" ) == 0 ) return kPlyI16;
    if ( strcmp( name, "ushort" ) == 0 || strcmp( name, "uint16" ) == 0 ) return kPlyU16;
    if ( strcmp( name, "int" ) == 0 || strcmp( name, "int32" ) == 0 ) return kPlyI32;
    if ( strcmp( name, "uint" ) == 0 || strcmp( name, "uint32" ) == 0 ) return kPlyU32;
    return kPlyUnknown;
}

/*
===================
===================
*/
static int ply_scalar_size( PlyScalar scalar ) {
    switch ( scalar ) {
        case kPlyI8:
        case kPlyU8: return 1;
        case kPlyI16:
        case kPlyU16: return 2;
        case kPlyI32:
        case kPlyU32:
        case kPlyF32: return 4;
        case kPlyF64: return 8;
        default: return 0;
    }
}

/*
===================
===================
*/
static float ply_read( const u8 * vertex, const PlyProperty * property ) {
    const u8 * at = vertex + property->offset;
    switch ( property->scalar ) {
        case kPlyF32: { float v; memcpy( &v, at, 4 ); return v; }
        case kPlyF64: { double v; memcpy( &v, at, 8 ); return float( v ); }
        case kPlyI8: return float( *(const i8 *) at );
        case kPlyU8: return float( *(const u8 *) at );
        case kPlyI16: { i16 v; memcpy( &v, at, 2 ); return float( v ); }
        case kPlyU16: { u16 v; memcpy( &v, at, 2 ); return float( v ); }
        case kPlyI32: { i32 v; memcpy( &v, at, 4 ); return float( v ); }
        case kPlyU32: { u32 v; memcpy( &v, at, 4 ); return float( v ); }
        default: return 0.0f;
    }
}

/*
===================
===================
*/
static const PlyProperty * find_property( const PlyProperty * properties, int count, const char * name ) {
    for ( int i = 0; i < count; i++ ) {
        if ( strcmp( properties[i].name, name ) == 0 ) {
            return &properties[i];
        }
    }
    return nullptr;
}

/*
===================
===================
*/
static void trim_line( char * line ) {
    // Trims the trailing newline, and the carriage return a file written on Windows leaves behind.
    int length = int( strlen( line ) );
    while ( length > 0 && ( line[length - 1] == '\n' || line[length - 1] == '\r' ) ) {
        line[--length] = '\0';
    }
}

/*
===================
===================
*/
bool ply_load_scene( const char * path, Scene * scene, bool flip_to_y_up ) {
    FILE * file = fopen( path, "rb" );
    if ( !file ) {
        printf( "ply: cannot open %s\n", path );
        return false;
    }

    char line[512];
    if ( !fgets( line, sizeof( line ), file ) ) {
        printf( "ply: %s is empty\n", path );
        fclose( file );
        return false;
    }

    trim_line( line );
    if ( strcmp( line, "ply" ) != 0 ) {
        printf( "ply: %s does not start with a ply magic\n", path );
        fclose( file );
        return false;
    }

    bool        binary_little_endian = false;
    bool        end_of_header = false;
    bool        in_vertex_element = false;
    i64         vertex_count = 0;
    PlyProperty properties[kMaxProperties];
    int         property_count = 0;
    int         stride = 0;

    while ( fgets( line, sizeof( line ), file ) ) {
        trim_line( line );

        if ( strcmp( line, "end_header" ) == 0 ) {
            end_of_header = true;
            break;
        }

        if ( strncmp( line, "format ", 7 ) == 0 ) {
            binary_little_endian = strstr( line, "binary_little_endian" ) != nullptr;
            continue;
        }

        if ( strncmp( line, "element ", 8 ) == 0 ) {
            char name[64] = {};
            long long count = 0;
            if ( sscanf( line, "element %63s %lld", name, &count ) == 2 ) {
                in_vertex_element = strcmp( name, "vertex" ) == 0;
                if ( in_vertex_element ) {
                    vertex_count = i64( count );
                }
            }
            continue;
        }

        if ( strncmp( line, "property ", 9 ) == 0 && in_vertex_element ) {
            if ( strncmp( line, "property list", 13 ) == 0 ) {
                printf( "ply: list properties in the vertex element are not supported\n" );
                fclose( file );
                return false;
            }

            char type[32] = {};
            char name[64] = {};
            if ( sscanf( line, "property %31s %63s", type, name ) != 2 ) {
                continue;
            }

            const PlyScalar scalar = ply_scalar_from_name( type );
            if ( scalar == kPlyUnknown ) {
                printf( "ply: unsupported property type '%s'\n", type );
                fclose( file );
                return false;
            }

            if ( property_count == kMaxProperties ) {
                printf( "ply: more than %d vertex properties\n", kMaxProperties );
                fclose( file );
                return false;
            }

            PlyProperty * property = &properties[property_count++];
            snprintf( property->name, sizeof( property->name ), "%s", name );
            property->scalar = scalar;
            property->offset = stride;
            stride += ply_scalar_size( scalar );
        }
    }

    if ( !end_of_header ) {
        printf( "ply: %s has no end_header\n", path );
        fclose( file );
        return false;
    }

    if ( !binary_little_endian ) {
        printf( "ply: only binary_little_endian is supported\n" );
        fclose( file );
        return false;
    }

    if ( vertex_count <= 0 || stride <= 0 ) {
        printf( "ply: %s declares no vertex data\n", path );
        fclose( file );
        return false;
    }

    // The renderer only uses the degree-0 term, so f_rest_* is read past rather than decoded.
    const PlyProperty * px = find_property( properties, property_count, "x" );
    const PlyProperty * py = find_property( properties, property_count, "y" );
    const PlyProperty * pz = find_property( properties, property_count, "z" );
    const PlyProperty * opacity = find_property( properties, property_count, "opacity" );

    const PlyProperty * scale[3] = {
        find_property( properties, property_count, "scale_0" ),
        find_property( properties, property_count, "scale_1" ),
        find_property( properties, property_count, "scale_2" ),
    };
    const PlyProperty * rotation[4] = {
        find_property( properties, property_count, "rot_0" ),
        find_property( properties, property_count, "rot_1" ),
        find_property( properties, property_count, "rot_2" ),
        find_property( properties, property_count, "rot_3" ),
    };
    const PlyProperty * colour[3] = {
        find_property( properties, property_count, "f_dc_0" ),
        find_property( properties, property_count, "f_dc_1" ),
        find_property( properties, property_count, "f_dc_2" ),
    };

    bool complete = px && py && pz && opacity;
    for ( int i = 0; i < 3; i++ ) {
        complete = complete && scale[i] && colour[i];
    }
    for ( int i = 0; i < 4; i++ ) {
        complete = complete && rotation[i];
    }

    if ( !complete ) {
        printf( "ply: %s is missing the gaussian splatting properties (x/scale_*/rot_*/opacity/f_dc_*)\n", path );
        fclose( file );
        return false;
    }

    u8 * chunk = (u8 *) malloc( (u64) kChunkVertices * u64( stride ) );
    if ( !chunk ) {
        printf( "ply: out of memory staging %d vertices\n", kChunkVertices );
        fclose( file );
        return false;
    }

    list_clear( scene->gaussians );
    list_reserve( scene->gaussians, i32( vertex_count ) );
    if ( scene->gaussians.cap < i32( vertex_count ) ) {
        printf( "ply: out of memory reserving %lld splats\n", (long long) vertex_count );
        free( chunk );
        fclose( file );
        return false;
    }

    i64 loaded = 0;
    while ( loaded < vertex_count ) {
        const i64 remaining = vertex_count - loaded;
        const int batch = int( remaining < kChunkVertices ? remaining : kChunkVertices );

        if ( fread( chunk, u64( stride ), u64( batch ), file ) != u64( batch ) ) {
            printf( "ply: %s ended after %lld of %lld vertices\n", path, (long long) loaded, (long long) vertex_count );
            break;
        }

        for ( int i = 0; i < batch; i++ ) {
            const u8 * vertex = chunk + u64( i ) * u64( stride );

            Gaussian g;
            g.position = glm::vec3( ply_read( vertex, px ), ply_read( vertex, py ), ply_read( vertex, pz ) );

            // Scales are stored as logs so training can keep them positive without a constraint.
            g.scale = glm::vec3( expf( ply_read( vertex, scale[0] ) ),
                                 expf( ply_read( vertex, scale[1] ) ),
                                 expf( ply_read( vertex, scale[2] ) ) );

            // rot_0..3 is w, x, y, z, which is also glm::quat's constructor order.
            const glm::quat q = glm::normalize( glm::quat( ply_read( vertex, rotation[0] ),
                                                           ply_read( vertex, rotation[1] ),
                                                           ply_read( vertex, rotation[2] ),
                                                           ply_read( vertex, rotation[3] ) ) );
            g.rotation = glm::mat3_cast( q );

            // Opacity is a logit, and the colour is an SH coefficient centred on zero rather than 0.5.
            const float alpha = 1.0f / ( 1.0f + expf( -ply_read( vertex, opacity ) ) );
            g.colour = glm::vec4( 0.5f + kShC0 * ply_read( vertex, colour[0] ),
                                  0.5f + kShC0 * ply_read( vertex, colour[1] ),
                                  0.5f + kShC0 * ply_read( vertex, colour[2] ),
                                  alpha );

            if ( flip_to_y_up ) {
                // Rotate 180 degrees about X: negate y and z of the position, and of every column of the
                // rotation. Covariance is (R S)(R S)^T, so pushing the flip through R flips the whole splat.
                g.position.y = -g.position.y;
                g.position.z = -g.position.z;
                for ( int c = 0; c < 3; c++ ) {
                    g.rotation[c].y = -g.rotation[c].y;
                    g.rotation[c].z = -g.rotation[c].z;
                }
            }

            list_add( scene->gaussians, g );
        }

        loaded += batch;
    }

    free( chunk );
    fclose( file );

    scene->gaussians_dirty = true;
    printf( "ply: loaded %d splats from %s (%d bytes per vertex)\n", scene->gaussians.count, path, stride );
    return scene->gaussians.count > 0;
}
