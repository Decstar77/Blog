#pragma once

#include "gs_defines.h"
#include "gs_splats.h"

bool cuda_render_init( u32 gl_texture );
void cuda_render_shutdown();
void cuda_render_frame( Scene * scene, int width, int height, float time );
