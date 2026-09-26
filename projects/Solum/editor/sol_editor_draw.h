// sol_editor_draw.h : turns the document and the tools' state into render
// streams. Nothing in here decides anything; it only draws what it is given.
#pragma once

#include "sol_brush.h"
#include "sol_camera.h"
#include "sol_editor_doc.h"
#include "sol_render.h"

namespace sol {

    // The fragment shader always applies its key light, so overlay vertices
    // carry the light's own direction as their normal. dot( n, l ) is then 1
    // and their colours come through exactly as written.
    constexpr Vec3 kOverlayNormal = { 0.41646f, 0.83291f, 0.36440f };

    // One stream's worth of vertices and batches, rebuilt from scratch whenever
    // what it shows changes and handed to the renderer in one go.
    struct StreamBuilder {
        List<StaticMeshVertex>  vertices;
        List<RenderBatch>       batches;
    };

    void            StreamClear( StreamBuilder & builder );
    void            StreamFree( StreamBuilder & builder );
    // Empty batches are dropped on the way out.
    void            StreamSubmit( StreamBuilder & builder, Renderer * r, RenderStreamId stream );
    // Opens a batch. Every vertex added until the next one opens lands in it.
    // The pointer is for setting the less common fields - texture, point size,
    // screen space - and is only good until the next batch is opened.
    RenderBatch *   StreamBatch( StreamBuilder & builder, RenderBatchKind kind, Vec4 tint, u32 viewMask );
    // Draws the vertices of the batch just filled a second time, another way,
    // without storing them twice.
    RenderBatch *   StreamRepeat( StreamBuilder & builder, RenderBatchKind kind, Vec4 tint, u32 viewMask );
    void            StreamVertex( StreamBuilder & builder, Vec3 position, Vec3 normal, Vec3 color, Vec2 uv );
    void            StreamLine( StreamBuilder & builder, Vec3 a, Vec3 b, Vec3 color );
    void            StreamPoint( StreamBuilder & builder, Vec3 position, Vec3 color );

    struct EditorTextureEntry {
        SmallString             material;
        // Null when the asset failed to load, so it is not retried every frame.
        RenderTextureHandle     handle;
    };

    // Materials are loaded the first time something draws with them.
    struct EditorTextures {
        List<EditorTextureEntry>    entries;
        // Worn by untextured faces and by materials that failed to load: a
        // one-unit grid, so every surface shows its own scale.
        RenderTextureHandle         devTexture;
    };

    bool                EditorTexturesInit( EditorTextures & textures, Renderer * r );
    void                EditorTexturesFree( EditorTextures & textures );
    RenderTextureHandle EditorTextureFor( EditorTextures & textures, Renderer * r, StringView material );

    // Every visible brush of the document plus the previews: faces in all
    // views, edges depth tested in the views in mask3D and drawn over
    // everything in the views in mask2D, where a wireframe is what the eye
    // needs. Previews draw as if selected, which is what they become.
    void    EditorBuildWorld( StreamBuilder & builder, const EditorDoc & doc, const Brush * previews, i32 previewCount,
                              EditorTextures & textures, Renderer * r, u32 mask3D, u32 mask2D );

    // --- pieces for the overlay, added to whatever batch is open -----------
    void    EditorDrawBrushEdges( StreamBuilder & builder, const Brush & brush, Vec3 color );
    void    EditorDrawFaceFill( StreamBuilder & builder, const Brush & brush, i32 face, Vec3 color );
    void    EditorDrawBrushFill( StreamBuilder & builder, const Brush & brush, Vec3 color );

    // An endless grid in an orthographic pane's own plane. Lines thin out by
    // powers of two until they are at least a few pixels apart, so the grid
    // stays readable at every zoom without changing the snap step.
    void    EditorDrawOrthoGrid( StreamBuilder & builder, const OrthoCamera & camera, i32 pixelWidth, i32 pixelHeight, f32 step, u32 viewMask );

    // Screen-space pieces, in pixels from the pane's top left corner, for a
    // batch opened with screenSpace set.
    void    EditorDrawScreenRect( StreamBuilder & builder, f32 x0, f32 y0, f32 x1, f32 y1, f32 paneWidth, f32 paneHeight, Vec3 color );
    // The world axes as the pane sees them, tucked into its bottom left corner.
    void    EditorDrawAxisTripod( StreamBuilder & builder, Vec3 right, Vec3 up, f32 paneWidth, f32 paneHeight );

} // namespace sol
