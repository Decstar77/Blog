#include "sol_editor_draw.h"
#include "sol_asset.h"

#include <cmath>
#include <cstdio>

namespace sol {

    // Selected brushes read red through their textures, the way the eye
    // expects a selection to in a brush editor; a selected face goes further
    // so it stands out from a selected brush.
    constexpr Vec4 kSelectedBrushTint = { 1.45f, 0.72f, 0.62f, 1.0f };
    constexpr Vec4 kSelectedFaceTint = { 1.7f, 0.5f, 0.4f, 1.0f };

    // Edges are drawn white and coloured by their batch's tint.
    constexpr Vec3 kEdgeWhite = { 1.0f, 1.0f, 1.0f };
    constexpr Vec4 kEdgeTint3D = { 0.07f, 0.07f, 0.09f, 1.0f };
    constexpr Vec4 kEdgeTint2D = { 0.58f, 0.60f, 0.66f, 1.0f };
    constexpr Vec4 kSelectedEdgeTint = { 1.0f, 0.28f, 0.22f, 1.0f };
    // Selected edges hidden behind something, drawn under the visible ones so
    // the whole selection can be seen through the walls in front of it.
    constexpr Vec4 kSelectedEdgeHiddenTint = { 0.45f, 0.12f, 0.10f, 1.0f };
    constexpr Vec4 kSelectedFaceEdgeTint = { 1.0f, 0.82f, 0.25f, 1.0f };

    // --- streams ------------------------------------------------------------

    void StreamClear( StreamBuilder & builder ) {
        ListClear( builder.vertices );
        ListClear( builder.batches );
    }

    void StreamFree( StreamBuilder & builder ) {
        ListFree( builder.vertices );
        ListFree( builder.batches );
    }

    void StreamSubmit( StreamBuilder & builder, Renderer * r, RenderStreamId stream ) {
        ListRemoveIf( builder.batches, []( const RenderBatch & batch ) { return batch.vertexCount <= 0; } );
        RendererSetStream( r, stream, builder.vertices.data, builder.vertices.count, builder.batches.data, builder.batches.count );
    }

    RenderBatch * StreamBatch( StreamBuilder & builder, RenderBatchKind kind, Vec4 tint, u32 viewMask ) {
        RenderBatch batch = {};
        batch.kind = kind;
        batch.firstVertex = builder.vertices.count;
        batch.tint = tint;
        batch.pointSize = 1.0f;
        batch.viewMask = viewMask;
        return ListAdd( builder.batches, batch );
    }

    RenderBatch * StreamRepeat( StreamBuilder & builder, RenderBatchKind kind, Vec4 tint, u32 viewMask ) {
        RenderBatch * last = ListLast( builder.batches );
        if( last == nullptr ) {
            return StreamBatch( builder, kind, tint, viewMask );
        }
        RenderBatch batch = *last;
        batch.kind = kind;
        batch.tint = tint;
        batch.viewMask = viewMask;
        return ListAdd( builder.batches, batch );
    }

    void StreamVertex( StreamBuilder & builder, Vec3 position, Vec3 normal, Vec3 color, Vec2 uv ) {
        StaticMeshVertex vertex = {};
        vertex.position = position;
        vertex.normal = normal;
        vertex.color = color;
        vertex.uv = uv;
        ListAdd( builder.vertices, vertex );

        RenderBatch * batch = ListLast( builder.batches );
        if( batch != nullptr ) {
            batch->vertexCount++;
        }
    }

    void StreamLine( StreamBuilder & builder, Vec3 a, Vec3 b, Vec3 color ) {
        StreamVertex( builder, a, kOverlayNormal, color, Vec2{} );
        StreamVertex( builder, b, kOverlayNormal, color, Vec2{} );
    }

    void StreamPoint( StreamBuilder & builder, Vec3 position, Vec3 color ) {
        StreamVertex( builder, position, kOverlayNormal, color, Vec2{} );
    }

    // --- textures -----------------------------------------------------------

    constexpr i32 kDevTextureSize = 128;

    bool EditorTexturesInit( EditorTextures & textures, Renderer * r ) {
        TextureAsset asset = {};
        asset.width = kDevTextureSize;
        asset.height = kDevTextureSize;
        asset.meta = TextureMetaDefault();
        asset.meta.format = TextureFormat_RGBA8_UNORM;
        asset.meta.filter = TextureFilter_Linear;
        asset.meta.wrap = TextureWrap_Repeat;
        ListResize( asset.pixels, kDevTextureSize * kDevTextureSize * 4 );

        // One repeat is one world unit: a dark line on the unit boundary, a
        // fainter one at every quarter.
        for( i32 y = 0; y < kDevTextureSize; y++ ) {
            for( i32 x = 0; x < kDevTextureSize; x++ ) {
                const bool edge = x == 0 || y == 0 || x == kDevTextureSize - 1 || y == kDevTextureSize - 1;
                const bool quarter = ( x % ( kDevTextureSize / 4 ) ) == 0 || ( y % ( kDevTextureSize / 4 ) ) == 0;
                u8 value = 176;
                if( edge ) {
                    value = 118;
                } else if( quarter ) {
                    value = 158;
                }
                u8 * pixel = &asset.pixels[( y * kDevTextureSize + x ) * 4];
                pixel[0] = value;
                pixel[1] = (u8)( value + 2 );
                pixel[2] = (u8)( value + 8 );
                pixel[3] = 255;
            }
        }

        textures.devTexture = RendererCreateTexture( r, asset );
        ListFree( asset.pixels );
        return !HandleIsNull( textures.devTexture );
    }

    void EditorTexturesFree( EditorTextures & textures ) {
        // The renderer owns the textures themselves and frees them on
        // shutdown; only the lookup table is ours.
        ListFree( textures.entries );
        textures = {};
    }

    RenderTextureHandle EditorTextureFor( EditorTextures & textures, Renderer * r, StringView material ) {
        if( material.count == 0 ) {
            return textures.devTexture;
        }

        for( i32 i = 0; i < textures.entries.count; i++ ) {
            if( StringEquals( textures.entries[i].material, material ) ) {
                const RenderTextureHandle handle = textures.entries[i].handle;
                return HandleIsNull( handle ) ? textures.devTexture : handle;
            }
        }

        EditorTextureEntry entry = {};
        StringSet( entry.material, material );

        FixedString<1024> path = {};
        StringAppend( path, SOLUM_ASSET_DIR );
        StringAppend( path, "/" );
        StringAppend( path, material );
        StringAppend( path, ".meta" );

        TextureAsset asset = {};
        if( TextureAssetLoad( path, &asset ) ) {
            entry.handle = RendererCreateTexture( r, asset );
            TextureAssetFree( &asset );
        }
        if( HandleIsNull( entry.handle ) ) {
            fprintf( stderr, "Material '%.*s' did not load; drawing it with the dev texture\n", material.count, material.data );
        }
        ListAdd( textures.entries, entry );
        return HandleIsNull( entry.handle ) ? textures.devTexture : entry.handle;
    }

    // --- world --------------------------------------------------------------

    enum FaceLook {
        FaceLook_Normal,
        FaceLook_SelectedBrush,
        FaceLook_SelectedFace,
        FaceLook_Count,
    };

    static FaceLook FaceLookOf( const Brush & brush, const BrushFace & face, bool preview ) {
        if( preview || ( brush.flags & BrushFlag_Selected ) ) {
            return FaceLook_SelectedBrush;
        }
        return ( face.flags & BrushFaceFlag_Selected ) ? FaceLook_SelectedFace : FaceLook_Normal;
    }

    static void AddFaceTriangles( StreamBuilder & builder, const Brush & brush, const BrushFace & face, Vec3 color ) {
        const Vec3 normal = face.plane.normal;
        const Vec3 a = brush.points[face.firstPoint];
        const Vec2 uvA = FaceTextureUv( face.texture, normal, a );
        for( i32 i = 1; i + 1 < face.pointCount; i++ ) {
            const Vec3 b = brush.points[face.firstPoint + i];
            const Vec3 c = brush.points[face.firstPoint + i + 1];
            StreamVertex( builder, a, normal, color, uvA );
            StreamVertex( builder, b, normal, color, FaceTextureUv( face.texture, normal, b ) );
            StreamVertex( builder, c, normal, color, FaceTextureUv( face.texture, normal, c ) );
        }
    }

    // Walks the doc's visible brushes and then the previews, which is the
    // order every pass below has to agree on.
    struct WorldBrushes {
        const EditorDoc *   doc;
        const Brush *       previews;
        i32                 previewCount;
    };

    static i32 WorldBrushCount( const WorldBrushes & world ) {
        return world.doc->map.brushes.count + world.previewCount;
    }

    static const Brush * WorldBrushAt( const WorldBrushes & world, i32 index, bool * outPreview ) {
        const i32 docCount = world.doc->map.brushes.count;
        *outPreview = index >= docCount;
        const Brush * brush = index < docCount ? &world.doc->map.brushes[index] : &world.previews[index - docCount];
        return ( brush->flags & BrushFlag_Hidden ) ? nullptr : brush;
    }

    void EditorBuildWorld( StreamBuilder & builder, const EditorDoc & doc, const Brush * previews, i32 previewCount,
                           EditorTextures & textures, Renderer * r, u32 mask3D, u32 mask2D ) {
        StreamClear( builder );
        const WorldBrushes world = { &doc, previews, previewCount };
        const i32 brushCount = WorldBrushCount( world );

        // Faces are drawn a material at a time, one batch per material and
        // look, since a batch binds exactly one texture. A map uses a handful
        // of materials, so finding them by linear search is cheap.
        List<SmallString> materials = {};
        for( i32 b = 0; b < brushCount; b++ ) {
            bool preview = false;
            const Brush * brush = WorldBrushAt( world, b, &preview );
            if( brush == nullptr ) {
                continue;
            }
            for( i32 f = 0; f < brush->faces.count; f++ ) {
                const SmallString & material = brush->faces[f].texture.material;
                bool seen = false;
                for( i32 m = 0; m < materials.count && !seen; m++ ) {
                    seen = StringEquals( materials[m], material );
                }
                if( !seen ) {
                    ListAdd( materials, material );
                }
            }
        }

        const Vec4 lookTints[FaceLook_Count] = { kNoTint, kSelectedBrushTint, kSelectedFaceTint };
        const Vec3 white = { 1.0f, 1.0f, 1.0f };

        for( i32 m = 0; m < materials.count; m++ ) {
            const RenderTextureHandle texture = EditorTextureFor( textures, r, materials[m] );
            for( i32 look = 0; look < FaceLook_Count; look++ ) {
                StreamBatch( builder, RenderBatch_Solid, lookTints[look], kRenderAllViews )->texture = texture;
                for( i32 b = 0; b < brushCount; b++ ) {
                    bool preview = false;
                    const Brush * brush = WorldBrushAt( world, b, &preview );
                    if( brush == nullptr ) {
                        continue;
                    }
                    for( i32 f = 0; f < brush->faces.count; f++ ) {
                        const BrushFace & face = brush->faces[f];
                        if( FaceLookOf( *brush, face, preview ) == look && StringEquals( face.texture.material, materials[m] ) ) {
                            AddFaceTriangles( builder, *brush, face, white );
                        }
                    }
                }
            }
        }
        ListFree( materials );

        List<Vec3> segments = {};

        // Unselected edges: thin and dark over their own faces in 3D, and
        // light over everything in 2D, which is a wireframe view.
        StreamBatch( builder, RenderBatch_Lines, kEdgeTint3D, mask3D );
        for( i32 b = 0; b < brushCount; b++ ) {
            bool preview = false;
            const Brush * brush = WorldBrushAt( world, b, &preview );
            if( brush == nullptr || preview || ( brush->flags & BrushFlag_Selected ) ) {
                continue;
            }
            BrushEdges( *brush, segments );
            for( i32 s = 0; s + 1 < segments.count; s += 2 ) {
                StreamLine( builder, segments[s], segments[s + 1], kEdgeWhite );
            }
        }
        StreamRepeat( builder, RenderBatch_LinesOnTop, kEdgeTint2D, mask2D );

        StreamBatch( builder, RenderBatch_LinesOnTop, kSelectedEdgeHiddenTint, mask3D );
        for( i32 b = 0; b < brushCount; b++ ) {
            bool preview = false;
            const Brush * brush = WorldBrushAt( world, b, &preview );
            if( brush == nullptr || !( preview || ( brush->flags & BrushFlag_Selected ) ) ) {
                continue;
            }
            BrushEdges( *brush, segments );
            for( i32 s = 0; s + 1 < segments.count; s += 2 ) {
                StreamLine( builder, segments[s], segments[s + 1], kEdgeWhite );
            }
        }
        StreamRepeat( builder, RenderBatch_Lines, kSelectedEdgeTint, mask3D );
        StreamRepeat( builder, RenderBatch_LinesOnTop, kSelectedEdgeTint, mask2D );

        StreamBatch( builder, RenderBatch_Lines, kSelectedFaceEdgeTint, mask3D );
        for( i32 b = 0; b < doc.map.brushes.count; b++ ) {
            const Brush & brush = doc.map.brushes[b];
            if( brush.flags & BrushFlag_Hidden ) {
                continue;
            }
            for( i32 f = 0; f < brush.faces.count; f++ ) {
                const BrushFace & face = brush.faces[f];
                if( !( face.flags & BrushFaceFlag_Selected ) ) {
                    continue;
                }
                for( i32 i = 0; i < face.pointCount; i++ ) {
                    StreamLine( builder, brush.points[face.firstPoint + i],
                                brush.points[face.firstPoint + ( i + 1 ) % face.pointCount], kEdgeWhite );
                }
            }
        }
        StreamRepeat( builder, RenderBatch_LinesOnTop, kSelectedFaceEdgeTint, mask2D );

        ListFree( segments );
    }

    // --- overlay pieces -----------------------------------------------------

    void EditorDrawBrushEdges( StreamBuilder & builder, const Brush & brush, Vec3 color ) {
        List<Vec3> segments = {};
        BrushEdges( brush, segments );
        for( i32 s = 0; s + 1 < segments.count; s += 2 ) {
            StreamLine( builder, segments[s], segments[s + 1], color );
        }
        ListFree( segments );
    }

    void EditorDrawFaceFill( StreamBuilder & builder, const Brush & brush, i32 face, Vec3 color ) {
        if( face < 0 || face >= brush.faces.count ) {
            return;
        }
        const BrushFace & f = brush.faces[face];
        const Vec3 a = brush.points[f.firstPoint];
        for( i32 i = 1; i + 1 < f.pointCount; i++ ) {
            StreamVertex( builder, a, kOverlayNormal, color, Vec2{} );
            StreamVertex( builder, brush.points[f.firstPoint + i], kOverlayNormal, color, Vec2{} );
            StreamVertex( builder, brush.points[f.firstPoint + i + 1], kOverlayNormal, color, Vec2{} );
        }
    }

    void EditorDrawBrushFill( StreamBuilder & builder, const Brush & brush, Vec3 color ) {
        for( i32 f = 0; f < brush.faces.count; f++ ) {
            EditorDrawFaceFill( builder, brush, f, color );
        }
    }

    static Vec3 AxisColor( Vec3 axis, f32 brightness ) {
        const f32 ax = fabsf( axis.x );
        const f32 ay = fabsf( axis.y );
        const f32 az = fabsf( axis.z );
        if( ax >= ay && ax >= az ) {
            return Vec3{ 0.9f, 0.25f, 0.25f } * brightness;
        }
        if( ay >= az ) {
            return Vec3{ 0.3f, 0.8f, 0.3f } * brightness;
        }
        return Vec3{ 0.3f, 0.45f, 0.95f } * brightness;
    }

    void EditorDrawOrthoGrid( StreamBuilder & builder, const OrthoCamera & camera, i32 pixelWidth, i32 pixelHeight, f32 step, u32 viewMask ) {
        if( pixelWidth <= 0 || pixelHeight <= 0 || step <= 0.0f ) {
            return;
        }

        Vec3 right = {};
        Vec3 up = {};
        Vec3 forward = {};
        OrthoCameraAxes( camera, &right, &up, &forward );

        const f32 halfHeight = camera.halfHeight;
        const f32 halfWidth = halfHeight * (f32)pixelWidth / (f32)pixelHeight;
        const f32 pixelsPerUnit = (f32)pixelHeight / ( 2.0f * halfHeight );

        f32 spacing = step;
        for( i32 i = 0; i < 32 && spacing * pixelsPerUnit < 7.0f; i++ ) {
            spacing *= 2.0f;
        }

        const f32 centerRight = Vec3Dot( camera.center, right );
        const f32 centerUp = Vec3Dot( camera.center, up );
        const f32 depth = Vec3Dot( camera.center, forward );
        const f32 minRight = centerRight - halfWidth;
        const f32 maxRight = centerRight + halfWidth;
        const f32 minUp = centerUp - halfHeight;
        const f32 maxUp = centerUp + halfHeight;

        const Vec3 minor = { 0.13f, 0.13f, 0.16f };
        const Vec3 major = { 0.22f, 0.22f, 0.27f };

        StreamBatch( builder, RenderBatch_LinesOnTop, kNoTint, viewMask );

        const i32 firstRight = (i32)floorf( minRight / spacing );
        const i32 lastRight = (i32)ceilf( maxRight / spacing );
        for( i32 k = firstRight; k <= lastRight; k++ ) {
            const f32 r = (f32)k * spacing;
            // The line where the right coordinate is zero runs along up, so it
            // takes up's axis colour.
            const Vec3 color = k == 0 ? AxisColor( up, 0.6f ) : ( ( k % 8 ) == 0 ? major : minor );
            StreamLine( builder, right * r + up * minUp + forward * depth, right * r + up * maxUp + forward * depth, color );
        }

        const i32 firstUp = (i32)floorf( minUp / spacing );
        const i32 lastUp = (i32)ceilf( maxUp / spacing );
        for( i32 k = firstUp; k <= lastUp; k++ ) {
            const f32 u = (f32)k * spacing;
            const Vec3 color = k == 0 ? AxisColor( right, 0.6f ) : ( ( k % 8 ) == 0 ? major : minor );
            StreamLine( builder, right * minRight + up * u + forward * depth, right * maxRight + up * u + forward * depth, color );
        }
    }

    static Vec3 ScreenToClip( f32 x, f32 y, f32 paneWidth, f32 paneHeight ) {
        // Clip +y is the top of the pane: the renderer's viewport is flipped.
        return Vec3{ 2.0f * x / paneWidth - 1.0f, 1.0f - 2.0f * y / paneHeight, 0.0f };
    }

    void EditorDrawScreenRect( StreamBuilder & builder, f32 x0, f32 y0, f32 x1, f32 y1, f32 paneWidth, f32 paneHeight, Vec3 color ) {
        if( paneWidth <= 0.0f || paneHeight <= 0.0f ) {
            return;
        }
        const Vec3 a = ScreenToClip( x0, y0, paneWidth, paneHeight );
        const Vec3 b = ScreenToClip( x1, y0, paneWidth, paneHeight );
        const Vec3 c = ScreenToClip( x1, y1, paneWidth, paneHeight );
        const Vec3 d = ScreenToClip( x0, y1, paneWidth, paneHeight );
        StreamLine( builder, a, b, color );
        StreamLine( builder, b, c, color );
        StreamLine( builder, c, d, color );
        StreamLine( builder, d, a, color );
    }

    void EditorDrawAxisTripod( StreamBuilder & builder, Vec3 right, Vec3 up, f32 paneWidth, f32 paneHeight ) {
        if( paneWidth <= 0.0f || paneHeight <= 0.0f ) {
            return;
        }
        const Vec3 axes[3] = { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } };
        const f32 originX = 30.0f;
        const f32 originY = paneHeight - 30.0f;
        const f32 length = 22.0f;

        for( i32 a = 0; a < 3; a++ ) {
            const f32 sx = Vec3Dot( axes[a], right );
            const f32 sy = Vec3Dot( axes[a], up );
            // An axis pointing straight out of the screen has nothing to show.
            if( sx * sx + sy * sy < 0.01f ) {
                continue;
            }
            StreamLine( builder, ScreenToClip( originX, originY, paneWidth, paneHeight ),
                        ScreenToClip( originX + sx * length, originY - sy * length, paneWidth, paneHeight ),
                        AxisColor( axes[a], 1.0f ) );
        }
    }

} // namespace sol
