#include "raylib.h"

#if defined( PLATFORM_WEB )
    #include <emscripten/emscripten.h>
#endif

#include <glm/glm.hpp>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/*
===================
===================
*/
struct OutputImage {
    int width;
    int height;
    glm::vec4 * pixels;
};

/*
===================
===================
*/
OutputImage CreateOutputImage( int w, int h ) {
    OutputImage image = {};
    image.width = w;
    image.height = h;
    int size = w * h * sizeof( float ) * 4;
    image.pixels = (glm::vec4 *) malloc( size );
    memset( image.pixels, 0, size );
    return image;
}

/*
===================
===================
*/
void ClearOutputImage( OutputImage & image ) {
    memset( image.pixels, 0, (size_t) image.width * image.height * sizeof( float ) * 4 );
}

/*
===================
===================
*/
void PutPixel( OutputImage & image, int x, int y, glm::vec4 c ) {
    int idx = y * image.width + x;
    image.pixels[idx] = c;
}

/*
===================
===================
*/
void AddPixel( OutputImage & image, int x, int y, glm::vec4 c ) {
    int idx = y * image.width + x;
    image.pixels[idx] += c;
}

/*
===================
===================
*/
glm::vec4 GibPixel( OutputImage & image, int x, int y ) {
    int idx = y * image.width + x;
    return image.pixels[idx];
}

/*
===================
===================
*/
struct Gaussian2D {
    glm::vec2 mu;
    glm::mat2 sigma;
    glm::vec4 color;
};

/*
===================
===================
*/
struct GaussianParams {
    glm::vec2 mu;
    float sizeX;    // standard deviation in pixels along the local x axis
    float sizeY;    // ... and along the local y axis
    float rotation; // radians, rotates those axes into the image
    glm::vec3 color;
    float weight; // peak amplitude
};

/*
===================
===================
*/
Gaussian2D BuildGaussian( const GaussianParams & p ) {
    float c = cosf( p.rotation );
    float s = sinf( p.rotation );
    glm::mat2 R = glm::mat2( c, s, -s, c ); // columns, so this rotates by +rotation
    glm::mat2 S = glm::mat2( p.sizeX * p.sizeX, 0.0f, 0.0f, p.sizeY * p.sizeY );

    Gaussian2D g = {};
    g.mu = p.mu;
    g.sigma = R * S * glm::transpose( R );
    g.color = glm::vec4( p.color * p.weight, p.weight );
    return g;
}

/*
===================
===================
*/
float MahalanobisSq( const Gaussian2D & g, glm::vec2 p ) {
    glm::vec2 diff = p - g.mu;
    return glm::dot( diff, glm::inverse( g.sigma ) * diff );
}

/*
===================
===================
*/
void SplatGaussian( OutputImage & image, const Gaussian2D & g ) {
    glm::mat2 sigmaInv = glm::inverse( g.sigma );

    // Extent along x and y is the sqrt of the diagonal whatever the rotation is.
    float radiusX = 4.0f * sqrtf( g.sigma[0][0] );
    float radiusY = 4.0f * sqrtf( g.sigma[1][1] );

    int minX = (int) floorf( g.mu.x - radiusX );
    int maxX = (int) ceilf( g.mu.x + radiusX );
    int minY = (int) floorf( g.mu.y - radiusY );
    int maxY = (int) ceilf( g.mu.y + radiusY );

    if ( minX < 0 ) {
        minX = 0;
    }
    if ( minY < 0 ) {
        minY = 0;
    }
    if ( maxX > image.width - 1 ) {
        maxX = image.width - 1;
    }
    if ( maxY > image.height - 1 ) {
        maxY = image.height - 1;
    }

    for ( int y = minY; y <= maxY; y++ ) {
        for ( int x = minX; x <= maxX; x++ ) {
            glm::vec2 p = glm::vec2( x, y );
            glm::vec2 diff = p - g.mu;
            float res = glm::exp( -0.5f * glm::dot( diff, sigmaInv * diff ) );
            AddPixel( image, x, y, g.color * res );
        }
    }
}

/*
===================================================
================= Scene
===================================================
*/

const int MAX_GAUSSIANS = 64;
const float MIN_SIZE = 4.0f;
const float MAX_SIZE = 220.0f;

static std::vector<GaussianParams> scene;
static int selected = -1;

static GaussianParams brush = {};

static bool sceneDirty = true;
static bool showOutlines = true;
static float exposure = 1.0f;

static int stateVersion = 0;

/*
===================
===================
*/
void ResetBrush( int screenWidth, int screenHeight ) {
    brush.mu = glm::vec2( screenWidth, screenHeight ) * 0.5f;
    brush.sizeX = 60.0f;
    brush.sizeY = 60.0f;
    brush.rotation = 0.0f;
    brush.color = glm::vec3( 1.0f, 0.0f, 1.0f );
    brush.weight = 1.0f;
}

/*
===================
===================
*/
void RenderScene( OutputImage & image ) {
    ClearOutputImage( image );
    for ( size_t i = 0; i < scene.size(); i++ ) {
        SplatGaussian( image, BuildGaussian( scene[i] ) );
    }
}

/*
===================
===================
*/
int PickGaussian( glm::vec2 p ) {
    for ( int i = (int) scene.size() - 1; i >= 0; i-- ) {
        if ( MahalanobisSq( BuildGaussian( scene[i] ), p ) <= 1.0f ) {
            return i;
        }
    }
    return -1;
}

/*
===================
===================
*/
int AddGaussian( glm::vec2 at ) {
    if ( (int) scene.size() >= MAX_GAUSSIANS ) {
        return -1;
    }
    GaussianParams p = brush;
    p.mu = at;
    scene.push_back( p );
    sceneDirty = true;
    stateVersion++;
    return (int) scene.size() - 1;
}

/*
===================
===================
*/
void SelectGaussian( int index ) {
    if ( index == selected ) {
        return;
    }
    selected = index;
    if ( selected >= 0 ) {
        // Adopt the selection's look, so the sliders describe what is
        // highlighted on screen rather than the last thing that was typed.
        glm::vec2 keep = brush.mu;
        brush = scene[selected];
        brush.mu = keep;
    }
    stateVersion++;
}

/*
===================
===================
*/
void DeleteSelected() {
    if ( selected < 0 ) {
        return;
    }
    scene.erase( scene.begin() + selected );
    selected = -1;
    sceneDirty = true;
    stateVersion++;
}

/*
===================
===================
*/
void ClearScene() {
    scene.clear();
    selected = -1;
    sceneDirty = true;
    stateVersion++;
}

/*
===================
===================
*/
void LoadDefaultScene( int screenWidth, int screenHeight ) {
    ClearScene();
    ResetBrush( screenWidth, screenHeight );

    brush.color = glm::vec3( 1.0f, 0.0f, 1.0f );
    AddGaussian( glm::vec2( screenWidth + 200, screenHeight - 200 ) / 2.0f );

    brush.color = glm::vec3( 1.0f, 1.0f, 0.0f );
    AddGaussian( glm::vec2( screenWidth - 200, screenHeight - 200 ) / 2.0f );

    brush.color = glm::vec3( 0.0f, 1.0f, 0.0f );
    AddGaussian( glm::vec2( screenWidth, screenHeight + 100 ) / 2.0f );

    ResetBrush( screenWidth, screenHeight );
    selected = -1;
    stateVersion++;
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
void HdrToBytes( const OutputImage & src, unsigned char * rgba, float exposureScale ) {
    int count = src.width * src.height;
    for ( int i = 0; i < count; i++ ) {
        const glm::vec4 & p = src.pixels[i];
        for ( int c = 0; c < 3; c++ ) {
            float v = p[c] * exposureScale;
            if ( v < 0.0f ) {
                v = 0.0f;
            }
            v = v / ( 1.0f + v );      // Reinhard
            v = powf( v, 1.0f / 2.2f ); // gamma
            rgba[i * 4 + c] = (unsigned char) ( v * 255.0f + 0.5f );
        }
        rgba[i * 4 + 3] = 255;
    }
}

/*
===================
===================
*/
Texture2D CreateDisplayTexture( int width, int height ) {
    Image img = GenImageColor( width, height, BLACK );
    ImageFormat( &img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 );
    Texture2D tex = LoadTextureFromImage( img ); // uploads to GPU
    UnloadImage( img );
    return tex;
}

/*
===================
===================
*/
void DrawGaussianOutline( const GaussianParams & p, Color tint ) {
    const int SEGMENTS = 64;
    float c = cosf( p.rotation );
    float s = sinf( p.rotation );

    Vector2 prev = {};
    for ( int i = 0; i <= SEGMENTS; i++ ) {
        float t = ( 2.0f * PI * i ) / SEGMENTS;
        float ex = p.sizeX * cosf( t );
        float ey = p.sizeY * sinf( t );
        Vector2 cur = { p.mu.x + ex * c - ey * s, p.mu.y + ex * s + ey * c };
        if ( i > 0 ) {
            DrawLineV( prev, cur, tint );
        }
        prev = cur;
    }
}

/*
===================
===================
*/
float ClampF( float v, float lo, float hi ) {
    if ( v < lo ) {
        return lo;
    }
    if ( v > hi ) {
        return hi;
    }
    return v;
}

/*
===================
===================
*/
void ApplyBrushToSelection() {
    if ( selected >= 0 ) {
        glm::vec2 keep = scene[selected].mu;
        scene[selected] = brush;
        scene[selected].mu = keep;
        sceneDirty = true;
    }
}

/*
===================================================
================= Web implementation
===================================================
*/

static glm::vec2 pointerNorm = glm::vec2( 0.0f, 0.0f );
static bool pointerDown = false;
static bool pointerWasDown = false;
static bool dragging = false;

#if defined( PLATFORM_WEB )

extern "C" {

// nx, ny are 0..1 across the canvas; the frame loop turns them into pixels.
EMSCRIPTEN_KEEPALIVE void BG_Pointer( float nx, float ny, int down ) {
    pointerNorm = glm::vec2( ClampF( nx, 0.0f, 1.0f ), ClampF( ny, 0.0f, 1.0f ) );
    pointerDown = ( down != 0 );
}

EMSCRIPTEN_KEEPALIVE void BG_SetSizeX( float v ) {
    brush.sizeX = ClampF( v, MIN_SIZE, MAX_SIZE );
    ApplyBrushToSelection();
}

EMSCRIPTEN_KEEPALIVE void BG_SetSizeY( float v ) {
    brush.sizeY = ClampF( v, MIN_SIZE, MAX_SIZE );
    ApplyBrushToSelection();
}

EMSCRIPTEN_KEEPALIVE void BG_SetRotation( float degrees ) {
    brush.rotation = degrees * ( PI / 180.0f );
    ApplyBrushToSelection();
}

EMSCRIPTEN_KEEPALIVE void BG_SetWeight( float v ) {
    brush.weight = ClampF( v, 0.02f, 3.0f );
    ApplyBrushToSelection();
}

EMSCRIPTEN_KEEPALIVE void BG_SetColor( float r, float g, float b ) {
    brush.color = glm::vec3( ClampF( r, 0.0f, 1.0f ), ClampF( g, 0.0f, 1.0f ), ClampF( b, 0.0f, 1.0f ) );
    ApplyBrushToSelection();
}

EMSCRIPTEN_KEEPALIVE void BG_SetExposure( float e ) {
    e = ClampF( e, 0.05f, 8.0f );
    if ( e != exposure ) {
        exposure = e;
        sceneDirty = true;
    }
}

EMSCRIPTEN_KEEPALIVE void BG_SetShowOutlines( int on ) {
    showOutlines = ( on != 0 );
}

EMSCRIPTEN_KEEPALIVE void BG_DeleteSelected() {
    DeleteSelected();
}

EMSCRIPTEN_KEEPALIVE void BG_Clear() {
    ClearScene();
}

EMSCRIPTEN_KEEPALIVE void BG_LoadDefaultScene() {
    LoadDefaultScene( GetScreenWidth(), GetScreenHeight() );
}

// Read back. A click on the canvas can change the selection, and the page has
// to follow that, so it watches the version and pulls the values it needs.
EMSCRIPTEN_KEEPALIVE int BG_Version() {
    return stateVersion;
}

EMSCRIPTEN_KEEPALIVE int BG_Count() {
    return (int) scene.size();
}

EMSCRIPTEN_KEEPALIVE int BG_Selected() {
    return selected;
}

EMSCRIPTEN_KEEPALIVE int BG_MaxCount() {
    return MAX_GAUSSIANS;
}

EMSCRIPTEN_KEEPALIVE float BG_GetSizeX() {
    return brush.sizeX;
}

EMSCRIPTEN_KEEPALIVE float BG_GetSizeY() {
    return brush.sizeY;
}

EMSCRIPTEN_KEEPALIVE float BG_GetRotation() {
    return brush.rotation * ( 180.0f / PI );
}

EMSCRIPTEN_KEEPALIVE float BG_GetWeight() {
    return brush.weight;
}

EMSCRIPTEN_KEEPALIVE float BG_GetColor( int channel ) {
    if ( channel < 0 || channel > 2 ) {
        return 0.0f;
    }
    return brush.color[channel];
}

// The covariance the splat actually integrates, exported rather than
// reimplemented in JS so the matrix the page prints is the one being used.
EMSCRIPTEN_KEEPALIVE float BG_GetSigma( int row, int col ) {
    if ( row < 0 || row > 1 || col < 0 || col > 1 ) {
        return 0.0f;
    }
    return BuildGaussian( brush ).sigma[col][row]; // glm is column major
}

} // extern "C"

#endif // PLATFORM_WEB

/*
===================================================
================= Main
===================================================
*/

static OutputImage image = {};
static Texture2D displayTex = {};
static unsigned char * displayBytes = nullptr;

static void UpdateDrawFrame();

int main() {
    const int screenWidth = 800;
    const int screenHeight = 800;
    InitWindow( screenWidth, screenHeight, "Basic Gaussian Splats" );

    image = CreateOutputImage( screenWidth, screenHeight );
    displayBytes = (unsigned char *) malloc( (size_t) screenWidth * screenHeight * 4 );
    displayTex = CreateDisplayTexture( screenWidth, screenHeight );

    LoadDefaultScene( screenWidth, screenHeight );

#if defined( PLATFORM_WEB )
    emscripten_set_main_loop( UpdateDrawFrame, 0, 1 ); // 0 = use requestAnimationFrame
#else
    SetTargetFPS( 60 );
    while ( !WindowShouldClose() ) {
        UpdateDrawFrame();
    }

    UnloadTexture( displayTex );
    CloseWindow();
    free( displayBytes );
    free( image.pixels );
#endif
    return 0;
}

static void UpdateDrawFrame() {
    int screenWidth = image.width;
    int screenHeight = image.height;

#if defined( PLATFORM_WEB )
    glm::vec2 mouse = glm::vec2( pointerNorm.x * screenWidth, pointerNorm.y * screenHeight );
#else
    Vector2 raw = GetMousePosition();
    glm::vec2 mouse = glm::vec2( raw.x, raw.y );
    pointerDown = IsMouseButtonDown( MOUSE_BUTTON_LEFT );

    if ( IsKeyPressed( KEY_DELETE ) || IsKeyPressed( KEY_BACKSPACE ) ) {
        DeleteSelected();
    }
    if ( IsKeyPressed( KEY_C ) ) {
        ClearScene();
    }
    if ( IsKeyPressed( KEY_R ) ) {
        LoadDefaultScene( screenWidth, screenHeight );
    }
#endif

    // A press either grabs the splat under the cursor or drops a new one, and
    // either way that is what the drag then moves. A hit has to win, otherwise
    // anything sitting under a splat could never be picked up again.
    if ( pointerDown && !pointerWasDown ) {
        int hit = PickGaussian( mouse );
        if ( hit < 0 ) {
            hit = AddGaussian( mouse );
        }
        SelectGaussian( hit );
        dragging = ( hit >= 0 );
    }
    if ( !pointerDown ) {
        dragging = false;
    }
    if ( dragging && selected >= 0 ) {
        glm::vec2 clamped = glm::vec2( ClampF( mouse.x, 0.0f, (float) screenWidth ),
            ClampF( mouse.y, 0.0f, (float) screenHeight ) );
        if ( clamped != scene[selected].mu ) {
            scene[selected].mu = clamped;
            sceneDirty = true;
        }
    }
    pointerWasDown = pointerDown;

    if ( sceneDirty ) {
        RenderScene( image );
        HdrToBytes( image, displayBytes, exposure );
        UpdateTexture( displayTex, displayBytes );
        sceneDirty = false;
    }

    BeginDrawing();
    ClearBackground( BLACK );
    DrawTexture( displayTex, 0, 0, WHITE );

    if ( showOutlines ) {
        for ( size_t i = 0; i < scene.size(); i++ ) {
            bool isSelected = ( (int) i == selected );
            DrawGaussianOutline( scene[i], isSelected ? WHITE : Color{ 255, 255, 255, 70 } );
            if ( isSelected ) {
                DrawCircleV( Vector2{ scene[i].mu.x, scene[i].mu.y }, 3.0f, WHITE );
            }
        }
    }

#if !defined( PLATFORM_WEB )
    // The browser build has the page's controls for all of this.
    DrawText( TextFormat( "%d splats  |  click to place, drag to move, DEL delete, C clear, R reset", (int) scene.size() ),
        10, 10, 14, Color{ 200, 200, 200, 180 } );
#endif

    EndDrawing();
}
