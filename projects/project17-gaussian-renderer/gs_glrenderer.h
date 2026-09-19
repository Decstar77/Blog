#pragma once

#include "gs_defines.h"

struct GsGlRenderer {
    u32 program;
    u32 vao;
    u32 target;
    i32 target_width;
    i32 target_height;
};

bool gs_glrenderer_init( GsGlRenderer * renderer );
void gs_glrenderer_shutdown( GsGlRenderer * renderer );
bool gs_glrenderer_resize_target( GsGlRenderer * renderer, int width, int height );
void gs_glrenderer_draw( const GsGlRenderer * renderer, int viewport_width, int viewport_height );
