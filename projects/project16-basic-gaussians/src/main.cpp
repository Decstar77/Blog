#include "raylib.h"

#include <glm/glm.hpp>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>

struct OutputImage {
    int         width;
    int         height;
    glm::vec4 * pixels;
};


OutputImage CreateOutputImage(int w, int h) {
    OutputImage image;
    image.width = w;
    image.height = h;
    int size = w * h * sizeof( float ) * 4; 
    image.pixels = ( glm::vec4 * ) malloc( size );
    memset( image.pixels, 0, size );
    return image;
}

void PutPixel(OutputImage& image, int x, int y, glm::vec4 c) {
    int idx = y * image.width + x;
    image.pixels[idx] = c;
}

glm::vec4 GibPixel(OutputImage& image, int x, int y) {
    int idx = y * image.width + x;
    return image.pixels[idx];
}

struct Gaussian2D {
    glm::vec2 mu;
    glm::mat2 sigma;
    glm::vec4 color;
};

void SplatGaussian(OutputImage& image, const Gaussian2D & g) {
    for ( int y = 0; y < image.height; y++ ) {
        for ( int x = 0; x < image.width; x++ ) {
            glm::vec2 p = glm::vec2( x, y );
            glm::vec2 diff = p - g.mu;
            glm::vec2 inv = glm::inverse( g.sigma ) * diff;
            float res = glm::exp( -0.5f * glm::dot(diff, inv) );
            if (res > 0.001f) {
                int a = 2;
            }
            glm::vec4 col = g.color * res;
            PutPixel( image, x, y, col );
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
    img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;

    Texture2D tex = LoadTextureFromImage( img ); // uploads to GPU
    return tex;
}


int main() {
    const int screenWidth = 800;
    const int screenHeight = 800;
    InitWindow( screenWidth, screenHeight, "SphericalHarmonics - SH projection explorer" );

    OutputImage image = CreateOutputImage( screenWidth, screenHeight );
    Gaussian2D g;
    g.color = glm::vec4( 1, 1, 1, 1 );
    g.mu = glm::vec2( screenWidth, screenHeight ) / 2.0f;
    g.sigma = glm::mat2( 200 );

    Texture2D tex = HdrToTexture( image );
    SplatGaussian( image, g );

    SetTargetFPS( 60 );
    while ( !WindowShouldClose() ) {
        BeginDrawing();
        ClearBackground( BLACK );
        DrawTexture( tex, 0, 0, WHITE );
        EndDrawing();
    }

    return 0;
}
