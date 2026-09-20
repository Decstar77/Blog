#include <glm/glm.hpp>

#include "cuda_render.h"
#include "gs_list.h"
#include "gs_glrenderer.h"
#include "gs_window.h"
#include "gs_splats.h"
#include "gs_scene.h"
#include "gs_ply.h"

#include <cstring>

// Take about blog
// EWA stands for Elliptical Weighted Average
// Ray-Gaussian integration
// 2D Gaussians / planar disks

int main( int argc, char ** argv ) {
    const char * ply_path = nullptr;
    bool flip_to_y_up = false;

    for ( int i = 1; i < argc; i++ ) {
        if ( strcmp( argv[i], "--flip-y" ) == 0 ) {
            flip_to_y_up = true;
        } else if ( ply_path == nullptr ) {
            ply_path = argv[i];
        }
    }

    // if ( ply_path == nullptr ) {
    //     ply_path = "C:/Projects/2025/Blog/data/splats/gpu/scene.ply";
    // }

    if ( !gs_window_create( 1280, 720, "Gaussian Renderer" ) ) {
        return 1;
    }

    GsGlRenderer renderer;
    if ( !gs_glrenderer_init( &renderer ) ) {
        gs_window_destroy();
        return 1;
    }

    // With a .ply on the command line, show that; otherwise fall back to the procedural demo scene.
    Scene scene = {};
    if ( ply_path && ply_load_scene( ply_path, &scene, flip_to_y_up ) ) {
        scene_frame_camera( &scene );
    } else {
        scene_build_demo( &scene );
    }

    while ( !gs_window_should_close() ) {
        GsInput input = {};
        gs_window_poll_input( &input );
        camera_update( &scene.camera, input, gs_window_delta_time() );

        int width, height;
        gs_window_framebuffer_size( &width, &height );

        if ( gs_glrenderer_resize_target( &renderer, width, height ) ) {
            cuda_render_init( renderer.target );
        }

        cuda_render_frame( &scene, renderer.target_width, renderer.target_height, gs_window_time() );

        gs_glrenderer_draw( &renderer, width, height );
        gs_window_present();
    }

    cuda_render_shutdown();
    gs_glrenderer_shutdown( &renderer );
    gs_window_destroy();
    return 0;
}
