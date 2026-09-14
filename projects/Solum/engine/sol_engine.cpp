// sol_engine.cpp : Defines the entry point for the application.
//

#include "sol_asset.h"
#include "sol_camera.h"
#include "sol_defines.h"
#include "sol_math.h"
#include "sol_render.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdio>

namespace sol {

    // The GLFW half of the camera controls: everything below turns polled
    // window state into a FlyCameraInput. The camera itself lives in
    // sol_camera.cpp so the Qt editor drives the identical code.
    struct GLFWLookState {
        bool    looking;
        f64     lastCursorX;
        f64     lastCursorY;
    };

    static FlyCameraInput GatherCameraInput( GLFWLookState * look, GLFWwindow * window ) {
        FlyCameraInput input = {};

        // Look only while the right button is held, so the cursor stays usable
        // the rest of the time.
        const bool wantLook = glfwGetMouseButton( window, GLFW_MOUSE_BUTTON_RIGHT ) == GLFW_PRESS;
        if( wantLook != look->looking ) {
            look->looking = wantLook;
            glfwSetInputMode( window, GLFW_CURSOR, wantLook ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL );
            // Re-seed on the press, otherwise the first frame sees the whole
            // gap since the last drag as one enormous delta.
            glfwGetCursorPos( window, &look->lastCursorX, &look->lastCursorY );
        }

        input.looking = look->looking;
        if( look->looking ) {
            f64 cursorX = 0.0;
            f64 cursorY = 0.0;
            glfwGetCursorPos( window, &cursorX, &cursorY );

            input.lookDeltaX = (f32)( cursorX - look->lastCursorX );
            input.lookDeltaY = (f32)( cursorY - look->lastCursorY );
            look->lastCursorX = cursorX;
            look->lastCursorY = cursorY;
        }

        input.forward = glfwGetKey( window, GLFW_KEY_W ) == GLFW_PRESS;
        input.back    = glfwGetKey( window, GLFW_KEY_S ) == GLFW_PRESS;
        input.right   = glfwGetKey( window, GLFW_KEY_D ) == GLFW_PRESS;
        input.left    = glfwGetKey( window, GLFW_KEY_A ) == GLFW_PRESS;
        input.up      = glfwGetKey( window, GLFW_KEY_SPACE ) == GLFW_PRESS;
        input.down    = glfwGetKey( window, GLFW_KEY_LEFT_CONTROL ) == GLFW_PRESS;
        input.fast    = glfwGetKey( window, GLFW_KEY_LEFT_SHIFT ) == GLFW_PRESS;
        return input;
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

    // Placeholder: there is no asset-path resolution yet, so this points
    // straight at a location on the dev machine rather than anything relative
    // to the build. Replace once assets resolve relative to the executable.
    static const char * const kBrickTextureMetaPath =
        "C:/Projects/2025/Blog/projects/Solum/assets/T_Bricks1_Color.meta";

    // Built in memory so the textured plane always has something to draw even
    // before a real brick asset exists on disk.
    static TextureAsset MakeCheckerboardFallback() {
        constexpr i32 kSize = 8;

        TextureAsset asset = {};
        asset.width = kSize;
        asset.height = kSize;
        asset.meta.format = TextureFormat_RGBA8_SRGB;
        asset.meta.filter = TextureFilter_Nearest;
        asset.meta.wrap = TextureWrap_Repeat;

        ListResize( asset.pixels, kSize * kSize * 4 );
        for( i32 y = 0; y < kSize; y++ ) {
            for( i32 x = 0; x < kSize; x++ ) {
                const bool magenta = ( ( x + y ) & 1 ) == 0;
                u8 * texel = &asset.pixels[( y * kSize + x ) * 4];
                texel[0] = magenta ? 255 : 0;
                texel[1] = 0;
                texel[2] = magenta ? 255 : 0;
                texel[3] = 255;
            }
        }
        return asset;
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

    if( started ) {
        sol::TextureAsset textureAsset = {};
        bool loadedFromDisk = sol::TextureAssetLoad( sol::kBrickTextureMetaPath, &textureAsset );
        if( !loadedFromDisk ) {
            fprintf( stderr, "Failed to load %s, falling back to a checkerboard\n",
                     sol::kBrickTextureMetaPath );
            textureAsset = sol::MakeCheckerboardFallback();
        }

        // The renderer owns the texture from here on, so there is nothing to
        // release at shutdown beyond shutting the renderer down.
        sol::RenderTextureHandle brickTexture = sol::RendererCreateTexture( &renderer, textureAsset );
        if( sol::HandleIsNull( brickTexture ) ) {
            fprintf( stderr, "Failed to create the brick/checkerboard render texture\n" );
        } else if( !sol::RendererAddTexturedPlane( &renderer, brickTexture,
                                                    sol::Vec3{ 0.0f, 0.0f, 0.0f }, 2.0f ) ) {
            fprintf( stderr, "Failed to add the textured plane\n" );
        }

        // The CPU-side copy is only needed for the upload above; the renderer
        // now owns a GPU-resident copy.
        if( loadedFromDisk ) {
            sol::TextureAssetFree( &textureAsset );
        } else {
            sol::ListFree( textureAsset.pixels );
        }
    }

    if( !started ) {
        fprintf( stderr, "Failed to start the renderer\n" );
        sol::RendererShutdown( &renderer );
        glfwDestroyWindow( window );
        glfwTerminate();
        return 1;
    }

    sol::FlyCamera camera = sol::FlyCameraDefault();
    sol::GLFWLookState look = {};

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

        sol::FlyCameraUpdate( &camera, sol::GatherCameraInput( &look, window ), dt );
        sol::RendererSetViewProjection( &renderer,
                                        sol::FlyCameraViewProjection( camera, width, height ) );

        sol::RendererDrawFrame( &renderer );
    }

    sol::RendererShutdown( &renderer );
    glfwDestroyWindow( window );
    glfwTerminate();
    return 0;
}
