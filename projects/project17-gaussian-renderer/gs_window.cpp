#include "gs_window.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

struct GsWindow {
    GLFWwindow *    handle;
    double          cursor_x;
    double          cursor_y;
    double          last_time;
    float           delta_time;
    bool            looking;
};

static GsWindow * main_window = nullptr;

GsWindow * gs_window_create( int width, int height, const char * title ) {
    if ( !glfwInit() ) {
        printf( "failed to init glfw\n" );
        return nullptr;
    }

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 6 );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

    GLFWwindow * handle = glfwCreateWindow( width, height, title, nullptr, nullptr );
    if ( !handle ) {
        printf( "failed to create window\n" );
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent( handle );
    if ( !gladLoadGL( glfwGetProcAddress ) ) {
        printf( "failed to load gl\n" );
        glfwDestroyWindow( handle );
        glfwTerminate();
        return nullptr;
    }
    glfwSwapInterval( 1 );

    printf( "gl %s\n", glGetString( GL_VERSION ) );

    GsWindow * window = (GsWindow *) malloc( sizeof( GsWindow ) );
    if ( !window ) {
        glfwDestroyWindow( handle );
        glfwTerminate();
        return nullptr;
    }
    window->handle = handle;
    window->cursor_x = 0.0;
    window->cursor_y = 0.0;
    window->last_time = glfwGetTime();
    window->delta_time = 0.0f;
    window->looking = false;
    main_window = window;
    return window;
}

void gs_window_destroy( GsWindow * window ) {
    if ( !window ) {
        return;
    }
    glfwDestroyWindow( window->handle );
    glfwTerminate();
    free( window );
}

bool gs_window_should_close( const GsWindow * window ) {
    return glfwWindowShouldClose( window->handle ) != 0;
}

void gs_window_framebuffer_size( const GsWindow * window, int * width, int * height ) {
    glfwGetFramebufferSize( window->handle, width, height );
}

void gs_window_present( GsWindow * window ) {
    glfwSwapBuffers( window->handle );
    glfwPollEvents();
}

void gs_window_poll_input( GsWindow * window, GsInput * input ) {
    *input = {};

    const double now = glfwGetTime();
    window->delta_time = (float) fmin( now - window->last_time, 0.1 );
    window->last_time = now;

    GLFWwindow * handle = window->handle;

    const bool look = glfwGetMouseButton( handle, GLFW_MOUSE_BUTTON_RIGHT ) == GLFW_PRESS;
    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos( handle, &x, &y );

    if ( look && !window->looking ) {
        glfwSetInputMode( handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED );
        if ( glfwRawMouseMotionSupported() ) {
            glfwSetInputMode( handle, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE );
        }
        glfwGetCursorPos( handle, &x, &y );
    } else if ( !look && window->looking ) {
        glfwSetInputMode( handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL );
    } else if ( look ) {
        input->mouse_dx = (float) ( x - window->cursor_x );
        input->mouse_dy = (float) ( y - window->cursor_y );
    }

    window->cursor_x = x;
    window->cursor_y = y;
    window->looking = look;

    if ( glfwGetKey( handle, GLFW_KEY_D ) == GLFW_PRESS ) input->move_right += 1.0f;
    if ( glfwGetKey( handle, GLFW_KEY_A ) == GLFW_PRESS ) input->move_right -= 1.0f;
    if ( glfwGetKey( handle, GLFW_KEY_W ) == GLFW_PRESS ) input->move_forward += 1.0f;
    if ( glfwGetKey( handle, GLFW_KEY_S ) == GLFW_PRESS ) input->move_forward -= 1.0f;
    if ( glfwGetKey( handle, GLFW_KEY_SPACE ) == GLFW_PRESS ) input->move_up += 1.0f;
    if ( glfwGetKey( handle, GLFW_KEY_LEFT_CONTROL ) == GLFW_PRESS ) input->move_up -= 1.0f;

    input->fast = glfwGetKey( handle, GLFW_KEY_LEFT_SHIFT ) == GLFW_PRESS || glfwGetKey( handle, GLFW_KEY_RIGHT_SHIFT ) == GLFW_PRESS;
    if ( glfwGetKey( handle, GLFW_KEY_ESCAPE ) == GLFW_PRESS ) {
        glfwSetWindowShouldClose( handle, GLFW_TRUE );
    }
}

float gs_window_delta_time() {
    return main_window->delta_time;
}

float gs_window_time( void ) {
    return (float) glfwGetTime();
}

float gs_window_aspect() {
    int width = 0;
    int height = 0;
    glfwGetWindowSize( main_window->handle, &width, &height );
    return width / height;
}
