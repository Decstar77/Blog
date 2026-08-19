#include "raylib.h"

#if defined( PLATFORM_WEB )
    #include <emscripten/emscripten.h>
#endif

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "tinyexr.h"

const float RECIP_PI = 1 / PI;
const float Constants[] = {
    sqrtf( RECIP_PI ) * 0.5f,
    sqrtf( 3 * RECIP_PI ) * 0.5f,
    sqrtf( 15 * RECIP_PI ) * 0.5f,
    sqrtf( 5 * RECIP_PI ) * 0.25f,
    sqrtf( 15 * RECIP_PI ) * 0.25f,
    sqrtf( 70 * RECIP_PI ) * 0.125f,
    sqrtf( 105 * RECIP_PI ) * 0.5f,
    sqrtf( 42 * RECIP_PI ) * 0.125f,
    sqrtf( 7 * RECIP_PI ) * 0.25f,
    sqrtf( 105 * RECIP_PI ) * 0.25f
};

static float y00( float x, float y, float z ) { return Constants[0]; }
static float y_11( float x, float y, float z ) { return Constants[1] * y; }
static float y01( float x, float y, float z ) { return Constants[1] * z; }
static float y11( float x, float y, float z ) { return Constants[1] * x; }
static float y_22( float x, float y, float z ) { return Constants[2] * y * x; }
static float y_12( float x, float y, float z ) { return Constants[2] * y * z; }
static float y02( float x, float y, float z ) { return Constants[3] * ( 3 * z * z - 1.0 ); }
static float y12( float x, float y, float z ) { return Constants[2] * x * z; }
static float y22( float x, float y, float z ) { return Constants[4] * ( x * x - y * y ); }
static float y_33( float x, float y, float z ) { return Constants[5] * y * ( 3 * x * x - y * y ); }
static float y_23( float x, float y, float z ) { return Constants[6] * z * ( y * x ); }
static float y_13( float x, float y, float z ) { return Constants[7] * y * ( 5 * z * z - 1 ); }
static float y03( float x, float y, float z ) { return Constants[8] * z * ( 5 * z * z - 3 ); }
static float y13( float x, float y, float z ) { return Constants[7] * x * ( 5 * z * z - 1 ); }
static float y23( float x, float y, float z ) { return Constants[9] * z * ( x * x - y * y ); }
static float y33( float x, float y, float z ) { return Constants[5] * x * ( x * x - 3 * y * y ); }

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Vec2 {
    float u;
    float v;
};

/*
===================
===================
*/
static void EvalSHBasis( Vec3 dir, int l, float * values, int & size ) {
    const float x = dir.x;
    const float y = dir.y;
    const float z = dir.z;
    switch ( l ) {
        case 0: {
            float value = y00( x, y, z );
            size = 1;
            values[0] = value;
        } break;
        case 1: {
            float storedValues[] = {
                y00( x, y, z ),  // l = 0
                y_11( x, y, z ), // l = 1
                y01( x, y, z ),
                y11( x, y, z )
            };
            size = 4;
            memcpy( values, storedValues, size * sizeof( float ) );
        }; break;
        case 2: {
            float storedValues[] = {
                y00( x, y, z ),  // l = 0
                y_11( x, y, z ), // l = 1
                y01( x, y, z ),
                y11( x, y, z ),
                y_22( x, y, z ), // l = 2
                y_12( x, y, z ),
                y02( x, y, z ),
                y12( x, y, z ),
                y22( x, y, z )
            };
            size = 9;
            memcpy( values, storedValues, size * sizeof( float ) );
        }; break;
        default: {
            float storedValues[] = {
                y00( x, y, z ),  // l = 0
                y_11( x, y, z ), // l = 1
                y01( x, y, z ),
                y11( x, y, z ),
                y_22( x, y, z ), // l = 2
                y_12( x, y, z ),
                y02( x, y, z ),
                y12( x, y, z ),
                y22( x, y, z ),
                y_33( x, y, z ), // l = 3
                y_23( x, y, z ),
                y_13( x, y, z ),
                y03( x, y, z ),
                y13( x, y, z ),
                y23( x, y, z ),
                y33( x, y, z )
            };
            size = 16;
            memcpy( values, storedValues, size * sizeof( float ) );
        } break;
    }
}

/*
===================
===================
*/
static Vec2 DirToEquirectUV( Vec3 dir ) {
    float phi = atan2f( dir.x, -dir.z );                         // range [-pi, pi]
    float theta = acosf( fmaxf( -1.0f, fminf( 1.0f, dir.y ) ) ); // Safer
    // float theta = acosf(dir.y); // range [0, pi] (0 = up, pi = down)

    float u = phi / ( 2.0f * (float) PI ) + 0.5f;
    float v = theta / (float) PI;

    return { u, v };
}

/*
===================
===================
*/
static Vec3 EquirectUVToDir( Vec2 uv ) {
    float phi = ( uv.u - 0.5f ) * 2.0f * (float) PI; // [-pi, pi]
    float theta = uv.v * (float) PI;                 // [0, pi]

    float sinTheta = sinf( theta );

    Vec3 dir;
    dir.x = sinTheta * sinf( phi );
    dir.y = cosf( theta );
    dir.z = -sinTheta * cosf( phi );

    return dir;
}

/*
===================
===================
*/
static Vec3 EquirectPixelToDir( int x, int y, int w, int h ) {
    Vec2 uv;
    uv.u = ( x + 0.5f ) / (float) w;
    uv.v = ( y + 0.5f ) / (float) h;

    return EquirectUVToDir( uv );
}

struct HdrImage {
    int     width;
    int     height;
    int     channels;
    float * pixels;
};

static HdrImage exrImage;
static HdrImage resImage;

/*
===================
===================
*/
static void ComputeUvIndices( const HdrImage & image, Vec2 uv, int & x, int & y ) {
    float fx = uv.u * image.width - 0.5f;
    float fy = uv.v * image.height - 0.5f;
    x = (int) floorf( fx );
    y = (int) floorf( fy );

    // wrap in longitude, clamp in latitude
    x = ( ( x % image.width ) + image.width ) % image.width;
    if ( y < 0 ) {
        y = 0;
    }
    if ( y > image.height - 1 ) {
        y = image.height - 1;
    }
}

/*
===================
===================
*/
static Vec3 Fetch( const HdrImage & image, Vec2 uv ) {
    int x;
    int y;
    ComputeUvIndices( image, uv, x, y );
    int idx = ( y * image.width + x ) * image.channels;
    return Vec3 { image.pixels[idx], image.pixels[idx + 1], image.pixels[idx + 2] };
}

/*
===================
===================
*/
static Vec3 Fetch( const HdrImage & image, int x, int y ) {
    int idx = ( y * image.width + x ) * image.channels;
    return Vec3 { image.pixels[idx], image.pixels[idx + 1], image.pixels[idx + 2] };
}

/*
===================
===================
*/
static void Place( HdrImage & image, int x, int y, Vec3 col ) {
    int idx = ( y * image.width + x ) * image.channels;
    image.pixels[idx] = col.x;
    image.pixels[idx + 1] = col.y;
    image.pixels[idx + 2] = col.z;
}

/*
===================
===================
*/
static void Test1() {
    for ( int y = 0; y < resImage.height; y++ ) {
        for ( int x = 0; x < resImage.width; x++ ) {
            Vec3 dir = EquirectPixelToDir( x, y, resImage.width, resImage.height );
            Vec2 uv = DirToEquirectUV( dir );
            Vec3 col = Fetch( exrImage, uv );
            Place( resImage, x, y, col );
        }
    }
}

/*
===================
===================
*/
static float AreaElement( float x, float y ) {
    return atan2( x * y, sqrt( x * x + y * y + 1 ) );
}

/*
===================
===================
*/
static float TexelCoordSolidAngle( float a_U, float a_V, float a_Size ) {
    float U = ( 2.0f * ( (float) a_U + 0.5f ) / (float) a_Size ) - 1.0f;
    float V = ( 2.0f * ( (float) a_V + 0.5f ) / (float) a_Size ) - 1.0f;
    float InvResolution = 1.0f / a_Size;
    float x0 = U - InvResolution;
    float y0 = V - InvResolution;
    float x1 = U + InvResolution;
    float y1 = V + InvResolution;
    float SolidAngle = AreaElement( x0, y0 ) - AreaElement( x0, y1 ) - AreaElement( x1, y0 ) + AreaElement( x1, y1 );

    return SolidAngle;
}

/*
===================
===================
*/
static Vec3 CubeUVToDir( int face, Vec2 uv ) {
    float sc = 2.0f * uv.u - 1.0f;
    float tc = 2.0f * uv.v - 1.0f;

    Vec3 dir = {};
    switch ( face ) {
        case 0:
            dir = { 1.0f, -tc, -sc };
            break; // +X
        case 1:
            dir = { -1.0f, -tc, sc };
            break; // -X
        case 2:
            dir = { sc, 1.0f, tc };
            break; // +Y
        case 3:
            dir = { sc, -1.0f, -tc };
            break; // -Y
        case 4:
            dir = { sc, -tc, 1.0f };
            break; // +Z
        case 5:
            dir = { -sc, -tc, -1.0f };
            break; // -Z
    }

    // Normalize
    float len = sqrtf( dir.x * dir.x + dir.y * dir.y + dir.z * dir.z );
    dir.x /= len;
    dir.y /= len;
    dir.z /= len;

    return dir;
}

/*
===================
===================
*/
static float Windowing( int l, float windowSize ) {
    if ( l == 0 ) {
        return 1.0f;
    }
    if ( float( l ) >= windowSize ) {
        return 0.0f; // past the first zero; the sinc's side lobes are not a window
    }
    const float theta = PI * float( l ) / windowSize;
    const float s = sinf( theta ) / theta;
    const float r = powf( s, 4 );
    return r;
}

struct SphereicalHarmonic {
    float r[16];
    float g[16];
    float b[16];
};

/*
===================
===================
*/
static void PrintSH( const SphereicalHarmonic & sh ) {
    std::cout << "R:";
    for ( int i = 0; i < 16; i++ ) {
        std::cout << "|" << sh.r[i];
    }
    std::cout << std::endl;

    std::cout << "G:";
    for ( int i = 0; i < 16; i++ ) {
        std::cout << "|" << sh.g[i];
    }
    std::cout << std::endl;

    std::cout << "B:";
    for ( int i = 0; i < 16; i++ ) {
        std::cout << "|" << sh.b[i];
    }
    std::cout << std::endl;
}

/*
===================
===================
*/
static void ComputeSHCube( SphereicalHarmonic & shResult, int l ) {
    shResult = {};

    const float CUBE_DIM = 128;
    const float TEXEL_SIZE = 1.0 / CUBE_DIM; // texel size in UV space.
    const float HALF_TEXEL_SIZE = 0.5 * TEXEL_SIZE;

    for ( int face = 0; face < 6; face++ ) {
        for ( int y = 0; y < CUBE_DIM; y++ ) {
            for ( int x = 0; x < CUBE_DIM; x++ ) {
                Vec2 cubeuv;
                cubeuv.u = ( x + 0.5f ) / CUBE_DIM;
                cubeuv.v = ( y + 0.5f ) / CUBE_DIM;

                const Vec3 dir = CubeUVToDir( face, cubeuv );
                const Vec2 equiuv = DirToEquirectUV( dir );
                const Vec3 col = Fetch( exrImage, equiuv );
                const float weight = TexelCoordSolidAngle( (float) x, (float) y, CUBE_DIM );
                int shSize = 0;
                float sh[16] = {};
                EvalSHBasis( dir, l, sh, shSize );

                float shRed[16] = {};
                float shGre[16] = {};
                float shBlu[16] = {};
                for ( int i = 0; i < shSize; i++ ) {
                    shRed[i] = sh[i] * weight * col.x;
                    shGre[i] = sh[i] * weight * col.y;
                    shBlu[i] = sh[i] * weight * col.z;
                }

                for ( int i = 0; i < shSize; i++ ) {
                    shResult.r[i] += shRed[i];
                    shResult.g[i] += shGre[i];
                    shResult.b[i] += shBlu[i];
                }
            }
        }
    }
}

/*
===================
===================
*/
static void ComputeSHEqui( SphereicalHarmonic & sh, int l ) {
    sh = {};

    const int w = exrImage.width;
    const int h = exrImage.height;
    const float dTheta = (float) PI / h;
    const float dPhi = 2.0f * (float) PI / w;

    for ( int y = 0; y < h; y++ ) {
        const float theta = ( y + 0.5f ) * dTheta;
        const float weight = sinf( theta ) * dTheta * dPhi;

        for ( int x = 0; x < w; x++ ) {
            const Vec3 dir = EquirectPixelToDir( x, y, w, h );
            const Vec3 col = Fetch( exrImage, x, y );

            int shSize = 0;
            float basis[16] = {};
            EvalSHBasis( dir, l, basis, shSize );

            for ( int i = 0; i < shSize; i++ ) {
                sh.r[i] += basis[i] * weight * col.x;
                sh.g[i] += basis[i] * weight * col.y;
                sh.b[i] += basis[i] * weight * col.z;
            }
        }
    }
}

/*
===================
===================
*/
static void ApplyWindowing( SphereicalHarmonic & sh, int lmax, float windowSize ) {
    const int count = ( lmax + 1 ) * ( lmax + 1 );
    for ( int i = 0; i < count; i++ ) {
        const int band = (int) sqrtf( (float) i ); // perfect squares are exact in float
        const float w = Windowing( band, windowSize );
        sh.r[i] *= w;
        sh.g[i] *= w;
        sh.b[i] *= w;
    }
}

/*
===================
===================
*/
static Vec3 SampleSh( const SphereicalHarmonic & coefs, Vec3 dir, int l ) {
    int basisSize = 0;
    float basis[16] = {};
    EvalSHBasis( dir, l, basis, basisSize );

    Vec3 result = {};
    for ( int i = 0; i < basisSize; i++ ) {
        result.x += coefs.r[i] * basis[i];
        result.y += coefs.g[i] * basis[i];
        result.z += coefs.b[i] * basis[i];
    }

    return result;
}

/*
===================
===================
*/
static void ShToHdrTexture( const SphereicalHarmonic & coef, int l ) {
    for ( int y = 0; y < resImage.height; y++ ) {
        for ( int x = 0; x < resImage.width; x++ ) {
            Vec3 dir = EquirectPixelToDir( x, y, resImage.width, resImage.height );
            Vec3 col = SampleSh( coef, dir, l );
            Vec2 uv = DirToEquirectUV( dir );
            Place( resImage, x, y, col );
        }
    }
}

/*
===================================================
================= Raylib drawing and implementation
===================================================
*/

/*
===================
===================
*/
Texture2D HdrToTexture( const HdrImage & src, float exposure ) {
    unsigned char * rgba = (unsigned char *) malloc( src.width * src.height * 4 );

    for ( int i = 0; i < src.width * src.height; i++ ) {
        const float * p = &src.pixels[i * src.channels];
        for ( int c = 0; c < 3; c++ ) {
            float v = p[c] * exposure;
            if ( v < 0.0f ) {
                v = 0.0f;
            }
            v = v / ( 1.0f + v ); // Reinhard
            v = powf( v, 1.0f / 2.2f );
            rgba[i * 4 + c] = (unsigned char) ( v * 255.0f + 0.5f );
        }
        rgba[i * 4 + 3] = 255;
    }

    Image img = {};
    img.data = rgba;
    img.width = src.width;
    img.height = src.height;
    img.mipmaps = 1;
    img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;

    Texture2D tex = LoadTextureFromImage( img ); // uploads to GPU
    UnloadImage( img );                          // frees rgba
    return tex;
}

const int MAX_L = 3;
enum ViewMode { VIEW_RAW = 0,
    VIEW_CUBE,
    VIEW_EQUI,
    VIEW_COUNT };
static const char * ViewName( int view ) {
    switch ( view ) {
        case VIEW_RAW:
            return "RAW HDR";
        case VIEW_CUBE:
            return "SH - cube projection (128^2 x 6)";
        default:
            return "SH - equirect projection";
    }
}

static const void * g_activeSlider = nullptr;
static bool Slider( Rectangle bounds, const char * label, float * value, float minV, float maxV, bool integral ) {
    const Vector2 mouse = GetMousePosition();

    if ( IsMouseButtonPressed( MOUSE_BUTTON_LEFT ) && CheckCollisionPointRec( mouse, bounds ) ) {
        g_activeSlider = value;
    }
    if ( !IsMouseButtonDown( MOUSE_BUTTON_LEFT ) && g_activeSlider == value ) {
        g_activeSlider = nullptr;
    }

    bool changed = false;
    if ( g_activeSlider == value ) {
        float t = ( mouse.x - bounds.x ) / bounds.width;
        t = fminf( 1.0f, fmaxf( 0.0f, t ) );
        float next = minV + t * ( maxV - minV );
        if ( integral ) {
            next = floorf( next + 0.5f );
        }
        if ( next != *value ) {
            *value = next;
            changed = true;
        }
    }

    const float t = ( *value - minV ) / ( maxV - minV );
    DrawRectangleRec( bounds, Fade( BLACK, 0.55f ) );
    DrawRectangle( (int) bounds.x, (int) bounds.y, (int) ( bounds.width * t ), (int) bounds.height, Fade( SKYBLUE, 0.55f ) );
    DrawRectangleLinesEx( bounds, 1.0f, GRAY );
    DrawRectangle( (int) ( bounds.x + bounds.width * t ) - 3, (int) bounds.y - 3, 6, (int) bounds.height + 6, RAYWHITE );
    DrawText( integral ? TextFormat( "%s %d", label, (int) *value )
                       : TextFormat( "%s %.1f", label, *value ),
        (int) ( bounds.x + bounds.width + 12 ), (int) bounds.y - 1, 18, RAYWHITE );
    return changed;
}

static void RebuildResult( const SphereicalHarmonic & raw, int l, float windowSize ) {
    SphereicalHarmonic sh = raw; // window a copy; the raw set is kept intact
    ApplyWindowing( sh, l, windowSize );
    ShToHdrTexture( sh, l );
}

// Frame state lives at file scope because the web build hands the loop body to
// the browser one frame at a time instead of running it inside main().
static SphereicalHarmonic shCubeRaw = {};
static SphereicalHarmonic shEquiRaw = {};
static float exposure = 1.0f;
static float lSlider = (float) MAX_L; // 0..3
static float windowSize = 6.0f;       // 3..10
static int view = VIEW_EQUI;
static bool shDirty = true;  // reconstruction is stale (view / l / windowSize changed)
static bool texDirty = true; // GPU textures are stale (exposure or resImage changed)
static Texture2D srcTex = {};
static Texture2D resTex = {};

static void UpdateDrawFrame();

#if defined( PLATFORM_WEB )
    // Preloaded into the Emscripten virtual FS by CMake; the 4k source is far
    // too heavy to ship over the wire and to project on a single main thread.
    #define HDR_PATH "malagas_ferry_road_1k.exr"
#else
    #define HDR_PATH "../malagas_ferry_road_1k.exr"
#endif

int main() {
    const char * err = nullptr;
    int ret = LoadEXR( &exrImage.pixels, &exrImage.width, &exrImage.height, HDR_PATH, &err );
    if ( ret != TINYEXR_SUCCESS ) {
        if ( err ) {
            std::cout << "Error loading exr";
            FreeEXRErrorMessage( err );
        }
        return -1;
    }

    exrImage.channels = 4; // LoadEXR always returns RGBA float

    resImage.width = exrImage.width;
    resImage.height = exrImage.height;
    resImage.channels = 3;
    resImage.pixels = (float *) malloc( (size_t) resImage.width * resImage.height * resImage.channels * sizeof( float ) );
    // Test1();

    // Project once at the highest band; the sliders only re-derive from these.
    ComputeSHCube( shCubeRaw, MAX_L );
    ComputeSHEqui( shEquiRaw, MAX_L );
    PrintSH( shCubeRaw );
    PrintSH( shEquiRaw );

    InitWindow( 1280, 720, "SphericalHarmonics - SH projection explorer" );

    srcTex = HdrToTexture( exrImage, exposure );

#if defined( PLATFORM_WEB )
    emscripten_set_main_loop( UpdateDrawFrame, 0, 1 ); // 0 = use requestAnimationFrame
#else
    SetTargetFPS( 60 );
    while ( !WindowShouldClose() ) {
        UpdateDrawFrame();
    }

    if ( resTex.id ) {
        UnloadTexture( resTex );
    }
    UnloadTexture( srcTex );
    CloseWindow();
    free( resImage.pixels );
    free( exrImage.pixels );
#endif
    return 0;
}

static void UpdateDrawFrame() {
    if ( IsKeyPressed( KEY_SPACE ) ) {
        view = ( view + 1 ) % VIEW_COUNT;
        shDirty = true;
    }

    const float oldExposure = exposure;
    if ( IsKeyDown( KEY_UP ) )
        exposure *= 1.02f;
    if ( IsKeyDown( KEY_DOWN ) )
        exposure /= 1.02f;
    if ( exposure != oldExposure ) {
        UnloadTexture( srcTex );
        srcTex = HdrToTexture( exrImage, exposure );
        texDirty = true;
    }

    const int l = (int) lSlider;

    // Reconstruct only when we are actually looking at an SH view; leaving
    // shDirty set means a slider moved in RAW is picked up on the way out.
    if ( view != VIEW_RAW && shDirty ) {
        RebuildResult( view == VIEW_CUBE ? shCubeRaw : shEquiRaw, l, windowSize );
        shDirty = false;
        texDirty = true;
    }
    if ( view != VIEW_RAW && texDirty ) {
        if ( resTex.id )
            UnloadTexture( resTex );
        resTex = HdrToTexture( resImage, exposure );
        texDirty = false;
    }

    const Texture2D tex = ( view == VIEW_RAW || !resTex.id ) ? srcTex : resTex;

    // fit the 2:1 equirect into the window, preserving aspect
    const float scale = fminf( GetScreenWidth() / (float) tex.width, GetScreenHeight() / (float) tex.height );
    const Rectangle srcRec = { 0, 0, (float) tex.width, (float) tex.height };
    const Rectangle dstRec = {
        ( GetScreenWidth() - tex.width * scale ) * 0.5f,
        ( GetScreenHeight() - tex.height * scale ) * 0.5f,
        tex.width * scale,
        tex.height * scale
    };

    // Per-band gains actually in effect, so the window is legible as numbers.
    char bandText[160];
    int off = snprintf( bandText, sizeof( bandText ), "band gain:" );
    for ( int b = 0; b <= l; b++ ) {
        off += snprintf( bandText + off, sizeof( bandText ) - off, "  %d:%.2f", b, Windowing( b, windowSize ) );
    }

    BeginDrawing();
    ClearBackground( BLACK );
    DrawTexturePro( tex, srcRec, dstRec, { 0, 0 }, 0.0f, WHITE );

    DrawRectangle( 10, 10, 470, 176, Fade( BLACK, 0.55f ) );
    DrawText( TextFormat( "%s   [SPACE]", ViewName( view ) ), 20, 18, 20, RAYWHITE );
    DrawText( TextFormat( "exposure %.2f   [UP/DOWN]", exposure ), 20, 44, 18, LIGHTGRAY );

    if ( Slider( { 20, 74, 220, 16 }, "l =", &lSlider, 0.0f, (float) MAX_L, true ) ) {
        shDirty = true;
    }
    if ( Slider( { 20, 104, 220, 16 }, "window =", &windowSize, 3.0f, 10.0f, false ) ) {
        shDirty = true;
    }

    DrawText( TextFormat( "%d coefficients", ( l + 1 ) * ( l + 1 ) ), 20, 130, 16, LIGHTGRAY );
    DrawText( bandText, 20, 150, 16, windowSize <= lSlider ? ORANGE : LIGHTGRAY );
    EndDrawing();
}
