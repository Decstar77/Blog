// sol_engine.cpp : Defines the entry point for the application.
//

#include "sol_defines.h"
#include "sol_math.h"
#include "sol_render.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdio>

namespace sol {

    // WASD + QE fly camera. Lives here rather than in the renderer: the renderer
    // only ever sees the matrix this produces.
    struct FlyCamera {
        Vec3    position;
        // Radians. yaw 0 / pitch 0 looks down -z, matching the math convention.
        f32     yaw;
        f32     pitch;
        f32     moveSpeed;      // units per second
        f32     lookSpeed;      // radians per pixel of mouse travel
        bool    looking;
        f64     lastCursorX;
        f64     lastCursorY;
    };

    constexpr f32 kPitchLimit = 89.0f * kDeg2Rad;

    static Vec3 FlyCameraForward( const FlyCamera & camera ) {
        const f32 cosPitch = cosf( camera.pitch );
        return Vec3{
            cosPitch * sinf( camera.yaw ),
            sinf( camera.pitch ),
            -cosPitch * cosf( camera.yaw ),
        };
    }

    static void FlyCameraUpdate( FlyCamera * camera, GLFWwindow * window, f32 dt ) {
        // Look only while the right button is held, so the cursor stays usable
        // the rest of the time.
        const bool wantLook = glfwGetMouseButton( window, GLFW_MOUSE_BUTTON_RIGHT ) == GLFW_PRESS;
        if( wantLook != camera->looking ) {
            camera->looking = wantLook;
            glfwSetInputMode( window, GLFW_CURSOR,
                              wantLook ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL );
            // Re-seed on the press, otherwise the first frame sees the whole
            // gap since the last drag as one enormous delta.
            glfwGetCursorPos( window, &camera->lastCursorX, &camera->lastCursorY );
        }

        if( camera->looking ) {
            f64 cursorX = 0.0;
            f64 cursorY = 0.0;
            glfwGetCursorPos( window, &cursorX, &cursorY );

            camera->yaw += (f32)( cursorX - camera->lastCursorX ) * camera->lookSpeed;
            camera->pitch -= (f32)( cursorY - camera->lastCursorY ) * camera->lookSpeed;
            camera->lastCursorX = cursorX;
            camera->lastCursorY = cursorY;

            // Stop short of straight up, where the forward vector and world up
            // line up and the right vector collapses.
            if( camera->pitch > kPitchLimit )  { camera->pitch = kPitchLimit; }
            if( camera->pitch < -kPitchLimit ) { camera->pitch = -kPitchLimit; }
        }

        const Vec3 forward = FlyCameraForward( *camera );
        const Vec3 right = Vec3Normalize( Vec3Cross( forward, Vec3{ 0.0f, 1.0f, 0.0f } ) );

        Vec3 move = {};
        if( glfwGetKey( window, GLFW_KEY_W ) == GLFW_PRESS ) { move = move + forward; }
        if( glfwGetKey( window, GLFW_KEY_S ) == GLFW_PRESS ) { move = move - forward; }
        if( glfwGetKey( window, GLFW_KEY_D ) == GLFW_PRESS ) { move = move + right; }
        if( glfwGetKey( window, GLFW_KEY_A ) == GLFW_PRESS ) { move = move - right; }
        // World up, not camera up: E and Q rise and fall regardless of pitch.
        if( glfwGetKey( window, GLFW_KEY_E ) == GLFW_PRESS ) { move.y += 1.0f; }
        if( glfwGetKey( window, GLFW_KEY_Q ) == GLFW_PRESS ) { move.y -= 1.0f; }

        f32 speed = camera->moveSpeed;
        if( glfwGetKey( window, GLFW_KEY_LEFT_SHIFT ) == GLFW_PRESS ) {
            speed *= 4.0f;
        }

        // Normalised so diagonals are not faster than the axes.
        if( Vec3Length( move ) > 0.0f ) {
            camera->position = camera->position + Vec3Normalize( move ) * ( speed * dt );
        }
    }

    static Mat4 FlyCameraViewProjection( const FlyCamera & camera, i32 width, i32 height ) {
        const Vec3 forward = FlyCameraForward( camera );
        const Mat4 view = Mat4LookAt( camera.position, camera.position + forward,
                                      Vec3{ 0.0f, 1.0f, 0.0f } );
        const f32 aspect = height > 0 ? (f32)width / (f32)height : 1.0f;
        const Mat4 projection = Mat4Perspective( 60.0f * kDeg2Rad, aspect, 0.1f, 1000.0f );
        return projection * view;
    }

    static void GLFWErrorCallback( int code, const char * description ) {
        fprintf( stderr, "GLFW error %d: %s\n", code, description );
    }

    static void GLFWKeyCallback( GLFWwindow * window, int key, int scancode, int action, int mods ) {
        SPLATS_UNUSED( scancode );
        SPLATS_UNUSED( mods );
        if( key == GLFW_KEY_ESCAPE && action == GLFW_PRESS ) {
            glfwSetWindowShouldClose( window, GLFW_TRUE );
        }
    }

    static void GLFWFramebufferSizeCallback( GLFWwindow * window, int width, int height ) {
        Renderer * renderer = (Renderer *)glfwGetWindowUserPointer( window );
        if( renderer != nullptr ) {
            RendererSetSize( renderer, width, height );
        }
    }

} // namespace sol

int main() {
    glfwSetErrorCallback( sol::GLFWErrorCallback );

    if( !glfwInit() ) {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        return 1;
    }

    if( !glfwVulkanSupported() ) {
        fprintf( stderr, "No Vulkan loader found\n" );
        glfwTerminate();
        return 1;
    }

    // The renderer owns the surface, so GLFW must not create a GL context.
    glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );

    GLFWwindow * window = glfwCreateWindow( 1280, 720, "Solum", nullptr, nullptr );
    if( window == nullptr ) {
        fprintf( stderr, "Failed to create window\n" );
        glfwTerminate();
        return 1;
    }

    static sol::Renderer renderer = {};
    glfwSetWindowUserPointer( window, &renderer );
    glfwSetKeyCallback( window, sol::GLFWKeyCallback );
    glfwSetFramebufferSizeCallback( window, sol::GLFWFramebufferSizeCallback );

    sol::u32 extCount = 0;
    const char ** extensions = glfwGetRequiredInstanceExtensions( &extCount );
    if( extensions == nullptr ) {
        fprintf( stderr, "GLFW reports no Vulkan surface support\n" );
        glfwDestroyWindow( window );
        glfwTerminate();
        return 1;
    }

    bool started = sol::RendererCreateInstance( &renderer, extensions, extCount );
    if( started ) {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if( glfwCreateWindowSurface( renderer.instance, window, nullptr, &surface ) != VK_SUCCESS ) {
            fprintf( stderr, "Failed to create a window surface\n" );
            started = false;
        } else {
            int width = 0;
            int height = 0;
            glfwGetFramebufferSize( window, &width, &height );
            started = sol::RendererStartup( &renderer, surface, true, width, height );
        }
    }

    if( started ) {
        sol::RendererAddDebugTriangle( &renderer );
    }

    if( !started ) {
        fprintf( stderr, "Failed to start the renderer\n" );
        sol::RendererShutdown( &renderer );
        glfwDestroyWindow( window );
        glfwTerminate();
        return 1;
    }

    sol::FlyCamera camera = {};
    // Backed off down +z so the debug triangle at the origin is in view.
    camera.position = sol::Vec3{ 0.0f, 0.0f, 2.0f };
    camera.moveSpeed = 3.0f;
    camera.lookSpeed = 0.0025f;

    sol::f64 lastTime = glfwGetTime();

    while( !glfwWindowShouldClose( window ) ) {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize( window, &width, &height );
        if( width == 0 || height == 0 ) {
            // Minimised: nothing to present, so sleep on the event queue.
            glfwWaitEvents();
            // The clock kept running while we were parked, so do not let that
            // turn into one huge step on the next frame.
            lastTime = glfwGetTime();
            continue;
        }

        glfwPollEvents();

        const sol::f64 now = glfwGetTime();
        const sol::f32 dt = (sol::f32)( now - lastTime );
        lastTime = now;

        sol::FlyCameraUpdate( &camera, window, dt );
        sol::RendererSetViewProjection( &renderer,
                                        sol::FlyCameraViewProjection( camera, width, height ) );

        sol::RendererDrawFrame( &renderer );
    }

    sol::RendererShutdown( &renderer );
    glfwDestroyWindow( window );
    glfwTerminate();
    return 0;
}
