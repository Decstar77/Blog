#pragma once

struct GsWindow;

// Per-frame input for the fly camera. Move axes are -1..1 in camera space
// (right / up / forward), mouse deltas are in pixels since the last poll and
// are only non-zero while the look button is held.
struct GsInput {
    float   move_right;
    float   move_up;
    float   move_forward;
    float   mouse_dx;
    float   mouse_dy;
    bool    fast;
};

GsWindow *  gs_window_create( int width, int height, const char * title );
void        gs_window_destroy( GsWindow * window );
bool        gs_window_should_close( const GsWindow * window );
void        gs_window_framebuffer_size( const GsWindow * window, int * width, int * height );
void        gs_window_present( GsWindow * window );
void        gs_window_poll_input( GsWindow * window, GsInput * input );
float       gs_window_time();
float       gs_window_delta_time();
float       gs_window_aspect();
