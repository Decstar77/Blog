#include <glm/glm.hpp>

#include "cuda_render.h"
#include "gs_list.h"
#include "gs_glrenderer.h"
#include "gs_window.h"
#include "gs_splats.h"
#include "gs_scene.h"

// Take about blog
// EWA stands for Elliptical Weighted Average
// Ray-Gaussian integration
// 2D Gaussians / planar disks

int main() {
    GsWindow * window = gs_window_create( 1280, 720, "Gaussian Renderer" );
    if ( !window ) {
        return 1;
    }

    GsGlRenderer renderer;
    if ( !gs_glrenderer_init( &renderer ) ) {
        gs_window_destroy( window );
        return 1;
    }

    Scene scene = {};
    scene_build_demo( &scene );

    while ( !gs_window_should_close( window ) ) {
        GsInput input = {};
        gs_window_poll_input( window, &input );
        camera_update( &scene.camera, input, gs_window_delta_time() );

        int width, height;
        gs_window_framebuffer_size( window, &width, &height );

        if ( gs_glrenderer_resize_target( &renderer, width, height ) ) {
            cuda_render_init( renderer.target );
        }

        cuda_render_frame( &scene, renderer.target_width, renderer.target_height, gs_window_time() );

        gs_glrenderer_draw( &renderer, width, height );
        gs_window_present( window );
    }

    cuda_render_shutdown();
    gs_glrenderer_shutdown( &renderer );
    gs_window_destroy( window );
    return 0;
}
