#pragma once
#include "sol_asset.h"
#include "sol_defines.h"
#include "sol_list.h"
#include "sol_math.h"

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

namespace sol {

    struct StaticMeshVertex {
        Vec3    position;
        Vec3    normal;
        Vec3    color;
        Vec2    uv;
    };

    // A GPU-resident, sampleable image plus the sampler that reads it. One of
    // these per loaded texture; the descriptor set is baked in at creation time
    // since nothing here ever changes which image a set points at.
    struct RenderTexture {
        VkImage             image;
        VmaAllocation       allocation;
        VkImageView         view;
        VkSampler           sampler;
        i32                 width;
        i32                 height;
        // Set 0, binding 0 of staticMeshPipelineLayout, already pointed at
        // view/sampler above. Allocated from Renderer::descriptorPool.
        VkDescriptorSet     descriptorSet;
    };

    // One indexed draw out of device-local memory. Every mesh the renderer draws
    // ends up as one of these, so the upload path and the pipeline are shared.
    struct RenderStaticMesh {
        VkBuffer        vertexBuffer;
        VmaAllocation   vertexAllocation;
        VkBuffer        indexBuffer;
        VmaAllocation   indexAllocation;
        i32             vertexCount;
        i32             indexCount;
        // Model to world. The renderer premultiplies the camera onto this before
        // pushing it to the vertex shader.
        Mat4            transform;
        // Every mesh must bind something here - Vulkan requires the combined
        // image sampler at set 0 binding 0 to be populated. Meshes that do not
        // care about a texture get the renderer's white 1x1 fallback.
        VkDescriptorSet textureSet;
    };

    // What a surface is made of, as far as the one pipeline currently cares:
    // a tint multiplied into every vertex, and the texture it samples.
    struct RenderMaterial {
        Vec3                    albedo;
        // Null falls back to the renderer's white 1x1, so the tint comes
        // through unmodified.
        const RenderTexture *   texture;
    };

    // Zeroing a RenderMaterial would give it a black albedo, so anything that
    // wants a sane default has to start here rather than at {}.
    RenderMaterial RenderMaterialDefault();

    // One rectangle of the surface, drawn with its own camera. The whole scene
    // is walked once per view, so a 3D pane and a top-down pane are two views
    // over the same mesh list rather than two renderers.
    struct RenderView {
        // Fraction of the swapchain, origin top left: { 0, 0, 1, 1 } is the
        // whole surface. Normalised rather than pixels so a resize needs no
        // work from whoever laid the views out.
        f32     x;
        f32     y;
        f32     width;
        f32     height;
        Mat4    viewProjection;
    };

    // Four is a full quad-view layout, which is as far as this needs to go.
    constexpr i32 kMaxRenderViews = 4;

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

        // Every buffer and image below is suballocated out of this rather than
        // getting its own vkAllocateMemory, which drivers cap at a few thousand.
        VmaAllocator                allocator;

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
        VmaAllocation               depthAllocation;
        VkImageView                 depthView;

        VkRenderPass                renderPass;
        VkCommandPool               commandPool;

        // Set 0 of staticMeshPipelineLayout: one combined image sampler, bound
        // per mesh before its draw call.
        VkDescriptorSetLayout       textureSetLayout;
        VkDescriptorPool            descriptorPool;
        // 1x1 opaque white, so a mesh with no real texture still satisfies the
        // descriptor requirement and renders as if unlit by any texture at all.
        RenderTexture               whiteTexture;

        // Viewport and scissor are dynamic, so a resize never rebuilds this.
        VkPipelineLayout            staticMeshPipelineLayout;
        VkPipeline                  staticMeshPipeline;
        // Same shaders, layout and vertex format as the mesh pipeline, built
        // with line topology and depth writes off. See CreateStaticMeshPipeline.
        VkPipeline                  gridPipeline;
        // World-space grid on the y = 0 plane, drawn in every view before the
        // meshes. Vertices only: a line list has nothing to index.
        VkBuffer                    gridVertexBuffer;
        VmaAllocation               gridVertexAllocation;
        i32                         gridVertexCount;
        bool                        gridVisible;
        // Drawn in order every frame. The renderer owns these and frees them on
        // shutdown.
        List<RenderStaticMesh>      staticMeshes;
        // Whoever owns the cameras sets these. Startup leaves one full-surface
        // view at identity, which draws meshes straight in clip space.
        RenderView                  views[kMaxRenderViews];
        i32                         viewCount;

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

    // Collapses the renderer to a single view covering the whole surface. The
    // shorthand for a shell that only ever shows one camera.
    void RendererSetViewProjection( Renderer * r, const Mat4 & viewProjection );

    // Replaces the set of views drawn each frame. Anything past kMaxRenderViews
    // is dropped.
    void RendererSetViews( Renderer * r, const RenderView * views, i32 count );

    // The grid is built at startup and shown by default; this hides it in every
    // view at once.
    void RendererSetGridVisible( Renderer * r, bool visible );

    void RendererDrawFrame( Renderer * r );

    // Uploads through a staging buffer, so the mesh lands in device-local memory.
    // Blocks until the copy is done - fine for load-time geometry, not for streaming.
    // texture may be null, in which case the mesh binds the renderer's white
    // fallback so its descriptor is still valid.
    bool RenderStaticMeshCreate( Renderer * r, const StaticMeshVertex * vertices, i32 vertexCount, const u32 * indices, i32 indexCount, const RenderTexture * texture, RenderStaticMesh * outMesh );
    void RenderStaticMeshDestroy( Renderer * r, RenderStaticMesh * mesh );

    // Hands the mesh to the renderer, which draws it every frame and owns it from
    // here on. Returns the stored copy, or nullptr if it could not be stored.
    RenderStaticMesh * RendererAddStaticMesh( Renderer * r, const RenderStaticMesh & mesh );

    // Placeholder geometry so there is something on screen. Delete once real
    // meshes are being loaded.
    bool RendererAddDebugTriangle( Renderer * r );

    // A two-triangle quad in the XY plane, facing +z, uvs spanning 0..1 across
    // it, vertex colour white so the sampled texel comes through unmodified.
    bool RendererAddTexturedPlane( Renderer * r, RenderTexture * texture, Vec3 center, f32 size );

    // Uploads asset.pixels through a staging buffer into a sampled, shader-read-
    // only-optimal image, and bakes a descriptor set pointing at it. Blocks
    // until the upload completes, same as RenderStaticMeshCreate.
    bool RenderTextureCreate( Renderer * r, const TextureAsset & asset, RenderTexture * outTexture );
    void RenderTextureDestroy( Renderer * r, RenderTexture * texture );

} // namespace sol
