#include "gs_window.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>

struct GsWindow {
    GLFWwindow * handle;
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

float gs_window_time( void ) {
    return (float) glfwGetTime();
}

float gs_window_aspect() {
    int width = 0;
    int height = 0;
    glfwGetWindowSize( main_window->handle, &width, &height );
    return width / height;
}
