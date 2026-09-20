#pragma once
struct GsInput {
    float   move_right;
    float   move_up;
    float   move_forward;
    float   mouse_dx;
    float   mouse_dy;
    bool    fast;
};

bool        gs_window_create( int width, int height, const char * title );
void        gs_window_destroy();
bool        gs_window_should_close();
void        gs_window_framebuffer_size( int * width, int * height );
void        gs_window_present();
void        gs_window_poll_input( GsInput * input );
float       gs_window_time();
float       gs_window_delta_time();
float       gs_window_aspect();
