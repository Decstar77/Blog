#pragma once
#include "sol_asset.h"
#include "sol_defines.h"
#include "sol_list.h"
#include "sol_math.h"
#include "sol_pool.h"

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

namespace sol {

    struct StaticMeshVertex {
        Vec3    position;
        Vec3    normal;
        Vec3    color;
        Vec2    uv;
    };

    // Pushed once per draw. Small enough to sit inside the 128 bytes every
    // Vulkan implementation guarantees, which is what lets the selection
    // highlight be a push rather than a rebuilt vertex buffer.
    struct StaticMeshPush {
        Mat4    mvp;
        Vec4    tint;
        // In pixels, and only read when the bound pipeline draws points. Every
        // other draw leaves it at 1. Pinned to 1 on a device without the
        // largePoints feature, where Vulkan allows no other value.
        f32     pointSize;
    };

    // Multiplied into the fragment colour, so this leaves it untouched.
    constexpr Vec4 kNoTint = { 1.0f, 1.0f, 1.0f, 1.0f };

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

    // How everything outside the renderer names a texture. The renderer owns
    // the RenderTexture itself; a null handle means "no texture", which the
    // draw path resolves to the white fallback.
    using RenderTextureHandle = Handle<RenderTexture>;

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
        // Multiplied into every fragment. kNoTint normally; the editor drives
        // this to highlight a selection without touching the mesh itself.
        Vec4            tint;
        // Resolved to a descriptor set at record time rather than baked in
        // here, so destroying or reloading a texture cannot leave a mesh
        // bound to a set that no longer points anywhere. A null handle - and
        // a stale one - lands on the renderer's white 1x1 fallback, which is
        // what keeps set 0 binding 0 populated the way Vulkan requires.
        RenderTextureHandle texture;
    };

    using RenderMeshHandle = Handle<RenderStaticMesh>;

    // What a surface is made of, as far as the one pipeline currently cares:
    // a tint multiplied into every vertex, and the texture it samples.
    struct RenderMaterial {
        Vec3                    albedo;
        // Null falls back to the renderer's white 1x1, so the tint comes
        // through unmodified.
        RenderTextureHandle     texture;
    };

    // Zeroing a RenderMaterial would give it a black albedo, so anything that
    // wants a sane default has to start here rather than at {}.
    RenderMaterial RenderMaterialDefault();

    // The edit cage for one primitive: its edges as a line list and its
    // vertices as a point list, both in the primitive's local space. Drawn
    // last in every view with the depth test off, so it reads as an overlay on
    // the shape rather than something competing with it for depth.
    struct RenderEditOverlay {
        VkBuffer        lineBuffer;
        VmaAllocation   lineAllocation;
        i32             lineVertexCount;
        VkBuffer        pointBuffer;
        VmaAllocation   pointAllocation;
        i32             pointVertexCount;
        // Local to world, matching the mesh the cage describes.
        Mat4            transform;
        bool            visible;
    };

    // A slice of the gizmo's vertex buffer drawn with its own tint, which is how
    // one static buffer carries handles that highlight independently.
    struct RenderGizmoRange {
        i32     firstVertex;
        i32     vertexCount;
        Vec4    tint;
    };

    constexpr i32 kMaxGizmoRanges = 8;

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
        // Every texture the renderer owns. Handed out as handles, so a texture
        // can be destroyed without anything still naming it reading freed
        // Vulkan objects.
        Pool<RenderTexture>         textures;
        // 1x1 opaque white, so a mesh with no real texture still satisfies the
        // descriptor requirement and renders as if unlit by any texture at all.
        RenderTextureHandle         whiteTexture;

        // Viewport and scissor are dynamic, so a resize never rebuilds this.
        VkPipelineLayout            staticMeshPipelineLayout;
        VkPipeline                  staticMeshPipeline;
        // Same shaders, layout and vertex format as the mesh pipeline, built
        // with line topology and depth writes off. See CreateStaticMeshPipeline.
        VkPipeline                  gridPipeline;
        // The edit cage, in line and point topology. Both go a step further
        // than the grid and turn the depth test off as well, so edges and
        // handles lying exactly on the surface they describe draw over it
        // instead of z-fighting with it.
        VkPipeline                  editLinePipeline;
        VkPipeline                  editPointPipeline;
        // False on a device without the largePoints feature, which pins vertex
        // handles to a single pixel - Vulkan permits no other size there.
        bool                        largePoints;
        RenderEditOverlay           editOverlay;

        // Static local-space geometry uploaded once; the transform and tints
        // are per frame, so following a moving object costs no GPU work.
        VkBuffer                    gizmoVertexBuffer;
        VmaAllocation               gizmoVertexAllocation;
        i32                         gizmoVertexCount;
        RenderGizmoRange            gizmoRanges[kMaxGizmoRanges];
        i32                         gizmoRangeCount;
        Mat4                        gizmoTransform;
        bool                        gizmoVisible;
        // World-space grid on the y = 0 plane, drawn in every view before the
        // meshes. Vertices only: a line list has nothing to index.
        VkBuffer                    gridVertexBuffer;
        VmaAllocation               gridVertexAllocation;
        i32                         gridVertexCount;
        bool                        gridVisible;
        // World units between lines. The grid keeps a constant line count and
        // grows its extent with this, so it stays useful at every zoom instead
        // of turning to mush when the spacing drops.
        f32                         gridSpacing;
        // Drawn in slot order every frame. The renderer owns these and frees
        // them on shutdown. A pool rather than a list because the world stores
        // references to individual meshes, and a list index stops naming the
        // same mesh the moment one is removed.
        Pool<RenderStaticMesh>      staticMeshes;
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

    // Rebuilds the grid at a new spacing. Idles the device first, so it is a
    // stall - fine for a key press, not for something driven per frame.
    bool RendererSetGridSpacing( Renderer * r, f32 spacing );

    // Replaces the edit cage. Both vertex lists are in the edited primitive's
    // local space and are drawn with transform. Either count may be zero.
    // Idles the device, same as RendererSetGridSpacing, so this belongs on a
    // selection change rather than in a frame.
    bool RendererSetEditOverlay( Renderer * r, const StaticMeshVertex * lineVertices, i32 lineVertexCount, const StaticMeshVertex * pointVertices, i32 pointVertexCount, const Mat4 & transform );

    // Takes the cage down and hands its buffers back. Cheap when there is no
    // cage up, so it is safe to call unconditionally.
    void RendererClearEditOverlay( Renderer * r );

    // Uploads the gizmo's local-space line geometry. Idles the device, so this
    // belongs at startup rather than in a frame.
    bool RendererSetGizmoGeometry( Renderer * r, const StaticMeshVertex * vertices, i32 vertexCount );

    // Where to draw that geometry and how to tint each slice of it. CPU state
    // only, so this is the per-frame call.
    void RendererSetGizmoDraw( Renderer * r, const Mat4 & transform, const RenderGizmoRange * ranges, i32 rangeCount );
    void RendererSetGizmoVisible( Renderer * r, bool visible );

    void RendererDrawFrame( Renderer * r );

    // Uploads through a staging buffer, so the mesh lands in device-local memory,
    // then hands it to the renderer, which draws it every frame and owns it from
    // here on. Blocks until the copy is done - fine for load-time geometry, not
    // for streaming. A null texture handle binds the white fallback.
    // Returns a null handle on failure.
    RenderMeshHandle RendererCreateStaticMesh( Renderer * r, const StaticMeshVertex * vertices, i32 vertexCount, const u32 * indices, i32 indexCount, RenderTextureHandle texture );

    // Frees the mesh's buffers and retires its slot, so every handle onto it
    // stops resolving. Idles the device first, since a frame in flight may
    // still be reading those buffers - a stall, same as RendererSetGridSpacing.
    void RendererDestroyStaticMesh( Renderer * r, RenderMeshHandle handle );

    // Null for a stale or null handle. The pointer is good until the next add
    // or remove on the mesh pool, so write through it and do not store it.
    RenderStaticMesh * RendererGetStaticMesh( Renderer * r, RenderMeshHandle handle );

    // Placeholder geometry so there is something on screen. Delete once real
    // meshes are being loaded.
    bool RendererAddDebugTriangle( Renderer * r );

    // A two-triangle quad in the XY plane, facing +z, uvs spanning 0..1 across
    // it, vertex colour white so the sampled texel comes through unmodified.
    bool RendererAddTexturedPlane( Renderer * r, RenderTextureHandle texture, Vec3 center, f32 size );

    // Uploads asset.pixels through a staging buffer into a sampled, shader-read-
    // only-optimal image, and bakes a descriptor set pointing at it. Blocks
    // until the upload completes, same as RendererCreateStaticMesh. The renderer
    // owns the result; callers keep only the handle. Null handle on failure.
    RenderTextureHandle RendererCreateTexture( Renderer * r, const TextureAsset & asset );

    // Meshes still naming this texture fall back to white rather than break.
    // Idles the device first, for the same reason the mesh teardown does.
    void RendererDestroyTexture( Renderer * r, RenderTextureHandle handle );

    // Null for a stale or null handle. Same lifetime caveat as the mesh getter.
    const RenderTexture * RendererGetTexture( const Renderer * r, RenderTextureHandle handle );

} // namespace sol
