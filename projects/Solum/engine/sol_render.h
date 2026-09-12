#pragma once
#include "sol_defines.h"
#include "sol_list.h"
#include "sol_math.h"

#include <vulkan/vulkan.h>

namespace sol {

    struct StaticMeshVertex {
        Vec3    position;
        Vec3    normal;
        Vec3    color;
    };

    // One indexed draw out of device-local memory. Every mesh the renderer draws
    // ends up as one of these, so the upload path and the pipeline are shared.
    struct RenderStaticMesh {
        VkBuffer        vertexBuffer;
        VkDeviceMemory  vertexMemory;
        VkBuffer        indexBuffer;
        VkDeviceMemory  indexMemory;
        i32             vertexCount;
        i32             indexCount;
        // Model to world. The renderer premultiplies the camera onto this before
        // pushing it to the vertex shader.
        Mat4            transform;
    };

    struct RenderMaterial {

    };

    // Two frames in flight: the CPU records frame N+1 while the GPU chews on N.
    constexpr i32 kFramesInFlight = 2;
    // Swapchains on desktop drivers hand back 2-4 images; 8 is slack.
    constexpr i32 kMaxSwapchainImages = 8;

    struct Renderer {
        VkInstance                  instance;
        VkDebugUtilsMessengerEXT    debugMessenger;
        VkSurfaceKHR                surface;
        // Qt hands out a surface it owns; GLFW hands out one we have to free.
        bool                        ownsSurface;

        VkPhysicalDevice            physicalDevice;
        u32                         graphicsFamily;
        u32                         presentFamily;

        VkDevice                    device;
        VkQueue                     graphicsQueue;
        VkQueue                     presentQueue;

        VkSwapchainKHR              swapchain;
        VkFormat                    swapchainFormat;
        VkExtent2D                  swapchainExtent;
        i32                         swapchainImageCount;
        VkImage                     swapchainImages[kMaxSwapchainImages];
        VkImageView                 swapchainImageViews[kMaxSwapchainImages];
        VkFramebuffer               framebuffers[kMaxSwapchainImages];
        // Present waits on this, so it has to be per image and not per frame:
        // the acquired image index is not the frame index.
        VkSemaphore                 renderFinished[kMaxSwapchainImages];
        VkFence                     imagesInFlight[kMaxSwapchainImages];

        // Sized to the swapchain, so it is torn down and rebuilt alongside it.
        // One shared depth buffer is enough: a frame's depth is never read after
        // its own render pass ends.
        VkFormat                    depthFormat;
        VkImage                     depthImage;
        VkDeviceMemory              depthMemory;
        VkImageView                 depthView;

        VkRenderPass                renderPass;
        VkCommandPool               commandPool;

        // Viewport and scissor are dynamic, so a resize never rebuilds this.
        VkPipelineLayout            staticMeshPipelineLayout;
        VkPipeline                  staticMeshPipeline;
        // Drawn in order every frame. The renderer owns these and frees them on
        // shutdown.
        List<RenderStaticMesh>      staticMeshes;
        // Whoever owns the camera sets this; the renderer just multiplies by it.
        // Identity until then, which draws meshes straight in clip space.
        Mat4                        viewProjection;

        VkCommandBuffer             commandBuffers[kFramesInFlight];
        VkSemaphore                 imageAvailable[kFramesInFlight];
        VkFence                     frameInFlight[kFramesInFlight];
        i32                         currentFrame;

        // Only consulted when the surface refuses to state its own extent.
        i32                         fallbackWidth;
        i32                         fallbackHeight;
        bool                        framebufferResized;
    };

    // Split from RendererStartup because Qt needs the VkInstance in hand before
    // it will create a window surface for us to start up against.
    bool RendererCreateInstance( Renderer * r, const char * const * requiredExtensions, u32 requiredCount );
    bool RendererStartup( Renderer * r, VkSurfaceKHR surface, bool ownsSurface, i32 width, i32 height );

    // Tears down everything that touches the surface, leaving the instance up.
    // Qt destroys its own surface when the window goes away, so the device has
    // to be idled and the swapchain dropped while that surface is still alive.
    void RendererShutdownDevice( Renderer * r );
    void RendererShutdown( Renderer * r );

    void RendererSetSize( Renderer * r, i32 width, i32 height );
    void RendererSetViewProjection( Renderer * r, const Mat4 & viewProjection );
    void RendererDrawFrame( Renderer * r );

    // Uploads through a staging buffer, so the mesh lands in device-local memory.
    // Blocks until the copy is done - fine for load-time geometry, not for streaming.
    bool RenderStaticMeshCreate( Renderer * r, const StaticMeshVertex * vertices, i32 vertexCount, const u32 * indices, i32 indexCount, RenderStaticMesh * outMesh );
    void RenderStaticMeshDestroy( Renderer * r, RenderStaticMesh * mesh );

    // Hands the mesh to the renderer, which draws it every frame and owns it from
    // here on. Returns the stored copy, or nullptr if it could not be stored.
    RenderStaticMesh * RendererAddStaticMesh( Renderer * r, const RenderStaticMesh & mesh );

    // Placeholder geometry so there is something on screen. Delete once real
    // meshes are being loaded.
    bool RendererAddDebugTriangle( Renderer * r );

} // namespace sol
