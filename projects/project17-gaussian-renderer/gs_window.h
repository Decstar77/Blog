#pragma once

struct GsWindow;

GsWindow *  gs_window_create( int width, int height, const char * title );
void        gs_window_destroy( GsWindow * window );
bool        gs_window_should_close( const GsWindow * window );
void        gs_window_framebuffer_size( const GsWindow * window, int * width, int * height );
void        gs_window_present( GsWindow * window );
float       gs_window_time();
float       gs_window_aspect();