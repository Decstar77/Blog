#include "raylib.h"

#include <glm/glm.hpp>
#include <string.h>

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
void SplatGaussian( OutputImage & image, const Gaussian2D & g ) {
    glm::mat2 sigmaInv = glm::inverse( g.sigma );
    for ( int y = 0; y < image.height; y++ ) {
        for ( int x = 0; x < image.width; x++ ) {
            glm::vec2 p = glm::vec2( x, y );
            glm::vec2 diff = p - g.mu;
            float res = glm::exp( -0.5f * glm::dot( diff, sigmaInv * diff ) );
            AddPixel( image, x, y, g.color * res );
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
Texture2D HdrToTexture( const OutputImage & src ) {
    Image img = {};
    img.data = src.pixels;
    img.width = src.width;
    img.height = src.height;
    img.mipmaps = 1;
    img.format = PIXELFORMAT_UNCOMPRESSED_R32G32B32A32;

    Texture2D tex = LoadTextureFromImage( img ); // uploads to GPU
    return tex;
}

int main() {
    const int screenWidth = 800;
    const int screenHeight = 800;
    InitWindow( screenWidth, screenHeight, "Basic Gaussian Splats" );

    OutputImage image = CreateOutputImage( screenWidth, screenHeight );

    Gaussian2D g1;
    g1.color = glm::vec4( 1, 0, 1, 1 );
    g1.mu = glm::vec2( screenWidth + 200, screenHeight - 200 ) / 2.0f;
    g1.sigma = glm::mat2( 60.0f * 60.0f );

    Gaussian2D g2 = g1;
    g2.color = glm::vec4( 1, 1, 0, 1 );
    g2.mu = glm::vec2( screenWidth - 200, screenHeight - 200 ) / 2.0f;
    
    Gaussian2D g3 = g1;
    g3.color = glm::vec4( 0, 1, 0, 1 );
    g3.mu = glm::vec2( screenWidth, screenHeight + 100 ) / 2.0f;

    SplatGaussian( image, g1 );
    SplatGaussian( image, g2 );
    SplatGaussian( image, g3 );

    Texture2D tex = HdrToTexture( image );

    SetTargetFPS( 60 );
    while ( !WindowShouldClose() ) {
        BeginDrawing();
        ClearBackground( BLACK );
        DrawTexture( tex, 0, 0, WHITE );
        EndDrawing();
    }

    return 0;
}
