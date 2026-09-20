#include "gs_window.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdio>

struct GsWindow {
    GLFWwindow *    handle;
    double          cursor_x;
    double          cursor_y;
    double          last_time;
    float           delta_time;
    bool            looking;
};

// There is only ever one window, so it lives here rather than on the heap. A zeroed handle is also the
// "not created yet" state.
static GsWindow main_window = {};

bool gs_window_create( int width, int height, const char * title ) {
    if ( !glfwInit() ) {
        printf( "failed to init glfw\n" );
        return false;
    }

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 6 );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

    GLFWwindow * handle = glfwCreateWindow( width, height, title, nullptr, nullptr );
    if ( !handle ) {
        printf( "failed to create window\n" );
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent( handle );
    if ( !gladLoadGL( glfwGetProcAddress ) ) {
        printf( "failed to load gl\n" );
        glfwDestroyWindow( handle );
        glfwTerminate();
        return false;
    }
    glfwSwapInterval( 1 );

    printf( "gl %s\n", glGetString( GL_VERSION ) );

    main_window = {};
    main_window.handle = handle;
    main_window.last_time = glfwGetTime();
    return true;
}

void gs_window_destroy() {
    if ( !main_window.handle ) {
        return;
    }

    glfwDestroyWindow( main_window.handle );
    glfwTerminate();
    main_window = {};
}

bool gs_window_should_close() {
    // Without a window there is nothing to keep a frame loop running.
    if ( !main_window.handle ) {
        return true;
    }
    return glfwWindowShouldClose( main_window.handle ) != 0;
}

void gs_window_framebuffer_size( int * width, int * height ) {
    glfwGetFramebufferSize( main_window.handle, width, height );
}

void gs_window_present() {
    glfwSwapBuffers( main_window.handle );
    glfwPollEvents();
}

void gs_window_poll_input( GsInput * input ) {
    *input = {};

    const double now = glfwGetTime();
    main_window.delta_time = (float) fmin( now - main_window.last_time, 0.1 );
    main_window.last_time = now;

    GLFWwindow * handle = main_window.handle;

    const bool look = glfwGetMouseButton( handle, GLFW_MOUSE_BUTTON_RIGHT ) == GLFW_PRESS;
    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos( handle, &x, &y );

    if ( look && !main_window.looking ) {
        glfwSetInputMode( handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED );
        if ( glfwRawMouseMotionSupported() ) {
            glfwSetInputMode( handle, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE );
        }
        glfwGetCursorPos( handle, &x, &y );
    } else if ( !look && main_window.looking ) {
        glfwSetInputMode( handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL );
    } else if ( look ) {
        input->mouse_dx = (float) ( x - main_window.cursor_x );
        input->mouse_dy = (float) ( y - main_window.cursor_y );
    }

    main_window.cursor_x = x;
    main_window.cursor_y = y;
    main_window.looking = look;

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
    return main_window.delta_time;
}

float gs_window_time() {
    return (float) glfwGetTime();
}

float gs_window_aspect() {
    int width = 0;
    int height = 0;
    glfwGetWindowSize( main_window.handle, &width, &height );
    return height > 0 ? float( width ) / float( height ) : 0.0f;
}
