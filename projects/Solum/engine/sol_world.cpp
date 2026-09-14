#include "sol_world.h"

#include <cstdio>

namespace sol {

    Transform TransformDefault() {
        Transform transform = {};
        transform.scale = Vec3{ 1.0f, 1.0f, 1.0f };
        return transform;
    }

    Mat4 TransformToMat4( const Transform & transform ) {
        return Mat4Translate( transform.position ) * Mat4FromEuler( transform.rotation ) *
               Mat4Scale( transform.scale );
    }

    World WorldCreate() {
        World world = {};
        world.selected = kNoPrimitive;
        return world;
    }

    RenderMeshHandle RenderMeshFromHalfMesh( Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material ) {
        List<HMTriVertex> triVertices = {};
        List<u32> triIndices = {};
        HalfMeshTriangulate( halfMesh, triVertices, triIndices );

        List<StaticMeshVertex> vertices = {};
        RenderMeshHandle handle = {};

        if( triVertices.count > 0 && triIndices.count > 0 ) {
            ListReserve( vertices, triVertices.count );
            for( i32 i = 0; i < triVertices.count; i++ ) {
                // The half-mesh side knows nothing about materials, so the tint
                // is folded in here rather than at triangulation time.
                StaticMeshVertex vertex = {};
                vertex.position = triVertices[i].position;
                vertex.normal = triVertices[i].normal;
                vertex.color = material.albedo;
                vertex.uv = triVertices[i].uv;
                ListAdd( vertices, vertex );
            }

            handle = RendererCreateStaticMesh( r, vertices.data, vertices.count,
                                               triIndices.data, triIndices.count, material.texture );
        }

        ListFree( triVertices );
        ListFree( triIndices );
        ListFree( vertices );
        return handle;
    }

    i32 WorldAddPrimitive( World & world, Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material, const Transform & transform ) {
        RenderMeshHandle renderMesh = RenderMeshFromHalfMesh( r, halfMesh, material );
        RenderStaticMesh * mesh = RendererGetStaticMesh( r, renderMesh );
        if( mesh == nullptr ) {
            return kNoPrimitive;
        }
        mesh->transform = TransformToMat4( transform );

        Primitive primitive = {};
        primitive.halfMesh = HalfMeshCopy( halfMesh );
        primitive.material = material;
        primitive.renderMesh = renderMesh;
        primitive.transform = transform;

        if( ListAdd( world.primitives, primitive ) == nullptr ) {
            HalfMeshFree( primitive.halfMesh );
            RendererDestroyStaticMesh( r, renderMesh );
            return kNoPrimitive;
        }

        return world.primitives.count - 1;
    }

    bool WorldRebuildPrimitive( World & world, Renderer * r, i32 primitive ) {
        if( primitive < 0 || primitive >= world.primitives.count ) {
            return false;
        }

        Primitive & entry = world.primitives[primitive];
        const RenderStaticMesh * previous = RendererGetStaticMesh( r, entry.renderMesh );
        if( previous == nullptr ) {
            return false;
        }

        // The transform comes from the primitive now, so only the tint has to
        // be carried across - dropping it would unhighlight a selected object.
        const Mat4 transform = TransformToMat4( entry.transform );
        const Vec4 tint = previous->tint;

        // Built the replacement first, so a failure here leaves the old mesh
        // on screen rather than a hole.
        RenderMeshHandle rebuilt = RenderMeshFromHalfMesh( r, entry.halfMesh, entry.material );
        RenderStaticMesh * mesh = RendererGetStaticMesh( r, rebuilt );
        if( mesh == nullptr ) {
            return false;
        }
        mesh->transform = transform;
        mesh->tint = tint;

        RendererDestroyStaticMesh( r, entry.renderMesh );
        entry.renderMesh = rebuilt;
        return true;
    }

    void WorldSetPrimitiveTransform( World & world, Renderer * r, i32 primitive, const Transform & transform ) {
        if( primitive < 0 || primitive >= world.primitives.count ) {
            return;
        }

        world.primitives[primitive].transform = transform;

        RenderStaticMesh * mesh = RendererGetStaticMesh( r, world.primitives[primitive].renderMesh );
        if( mesh == nullptr ) {
            return;
        }

        // Read fresh on the CPU when the next frame records, so there is
        // nothing to synchronise against here.
        mesh->transform = TransformToMat4( transform );
    }

    bool WorldGetPrimitiveTransform( const World & world, i32 primitive, Transform * outTransform ) {
        if( primitive < 0 || primitive >= world.primitives.count ) {
            return false;
        }

        if( outTransform != nullptr ) {
            *outTransform = world.primitives[primitive].transform;
        }
        return true;
    }

    // A plane is a zero-thickness box, and floating point can turn that slab
    // into a miss. A hair of padding keeps flat things clickable.
    constexpr f32 kPickPadding = 0.001f;

    bool WorldPick( const World & world, Renderer * r, Vec3 rayOrigin, Vec3 rayDirection,
                    i32 * outPrimitive ) {
        i32 best = kNoPrimitive;
        f32 bestDistance = 0.0f;

        for( i32 i = 0; i < world.primitives.count; i++ ) {
            const Primitive & primitive = world.primitives[i];
            const RenderStaticMesh * mesh = RendererGetStaticMesh( r, primitive.renderMesh );
            if( mesh == nullptr ) {
                continue;
            }
            if( primitive.halfMesh.vertices.count == 0 ) {
                continue;
            }

            const Mat4 & transform = mesh->transform;

            // Bounds taken from the transformed vertices rather than from a
            // transformed local box: exact whatever the transform does, where
            // moving a box's corners is only exact without rotation.
            Vec3 boundsMin = Mat4MulPoint( transform, primitive.halfMesh.vertices[0].position );
            Vec3 boundsMax = boundsMin;
            for( i32 v = 1; v < primitive.halfMesh.vertices.count; v++ ) {
                const Vec3 point = Mat4MulPoint( transform, primitive.halfMesh.vertices[v].position );
                boundsMin.x = Min( boundsMin.x, point.x );
                boundsMin.y = Min( boundsMin.y, point.y );
                boundsMin.z = Min( boundsMin.z, point.z );
                boundsMax.x = Max( boundsMax.x, point.x );
                boundsMax.y = Max( boundsMax.y, point.y );
                boundsMax.z = Max( boundsMax.z, point.z );
            }

            const Vec3 padding = { kPickPadding, kPickPadding, kPickPadding };
            boundsMin = boundsMin - padding;
            boundsMax = boundsMax + padding;

            f32 distance = 0.0f;
            if( !RayAabbIntersect( rayOrigin, rayDirection, boundsMin, boundsMax, &distance ) ) {
                continue;
            }

            if( best == kNoPrimitive || distance < bestDistance ) {
                best = i;
                bestDistance = distance;
            }
        }

        if( outPrimitive != nullptr ) {
            *outPrimitive = best;
        }
        return best != kNoPrimitive;
    }

    void WorldSetSelected( World & world, Renderer * r, i32 primitive ) {
        if( world.selected == primitive ) {
            return;
        }

        // Clear the old highlight before moving on, or it stays lit forever.
        if( world.selected >= 0 && world.selected < world.primitives.count ) {
            RenderStaticMesh * previous = RendererGetStaticMesh( r, world.primitives[world.selected].renderMesh );
            if( previous != nullptr ) {
                previous->tint = kNoTint;
            }
        }

        world.selected = primitive;

        if( primitive >= 0 && primitive < world.primitives.count ) {
            RenderStaticMesh * current = RendererGetStaticMesh( r, world.primitives[primitive].renderMesh );
            if( current != nullptr ) {
                current->tint = kSelectionTint;
            }
        }
    }

    void WorldSetPrimitiveHighlight( World & world, Renderer * r, i32 primitive, bool highlight ) {
        // Anything that is not the selection has no highlight either way, so
        // there is nothing here to drop and nothing to hand back.
        if( world.selected != primitive ) {
            return;
        }
        if( primitive < 0 || primitive >= world.primitives.count ) {
            return;
        }

        RenderStaticMesh * mesh = RendererGetStaticMesh( r, world.primitives[primitive].renderMesh );
        if( mesh == nullptr ) {
            return;
        }

        mesh->tint = highlight ? kSelectionTint : kNoTint;
    }

    // The fragment shader always applies its key light, so cage vertices carry
    // the light's own direction as their normal. dot( n, l ) is then 1 and the
    // colours below come through exactly as written, with no shading to read
    // as a handle being a different colour from its neighbour.
    constexpr Vec3 kEditOverlayNormal = { 0.41646f, 0.83291f, 0.36440f };
    constexpr Vec3 kEditEdgeColor = { 0.88f, 0.89f, 0.91f };
    constexpr Vec3 kEditVertexColor = { 0.09f, 0.16f, 0.42f };

    static void EditOverlayPushVertex( List<StaticMeshVertex> & vertices, Vec3 position, Vec3 color ) {
        StaticMeshVertex vertex = {};
        vertex.position = position;
        vertex.normal = kEditOverlayNormal;
        vertex.color = color;
        // Sampled against the white 1x1, so any uv gives the same texel.
        vertex.uv = Vec2{ 0.0f, 0.0f };
        ListAdd( vertices, vertex );
    }

    bool WorldSetEditOverlay( World & world, Renderer * r, i32 primitive ) {
        if( primitive < 0 || primitive >= world.primitives.count ) {
            return false;
        }

        const Primitive & entry = world.primitives[primitive];
        // The cage is drawn in local space, so it needs the same transform the
        // mesh is drawn with rather than one of its own.
        const RenderStaticMesh * mesh = RendererGetStaticMesh( r, entry.renderMesh );
        if( mesh == nullptr ) {
            return false;
        }

        const HalfMesh & halfMesh = entry.halfMesh;
        List<StaticMeshVertex> lines = {};
        List<StaticMeshVertex> points = {};

        // One line per edge, not per half-edge: the pair describes the same
        // segment, so walking edges draws each one once.
        for( i32 e = 0; e < halfMesh.edges.count; e++ ) {
            const i32 halfEdge = halfMesh.edges[e].halfEdge;
            const i32 from = halfMesh.halfEdges[halfEdge].vert;
            const i32 to = HalfEdgeDest( halfMesh, halfEdge );
            EditOverlayPushVertex( lines, halfMesh.vertices[from].position, kEditEdgeColor );
            EditOverlayPushVertex( lines, halfMesh.vertices[to].position, kEditEdgeColor );
        }

        for( i32 v = 0; v < halfMesh.vertices.count; v++ ) {
            EditOverlayPushVertex( points, halfMesh.vertices[v].position, kEditVertexColor );
        }

        const bool ok = RendererSetEditOverlay( r, lines.data, lines.count,
                                                points.data, points.count, mesh->transform );
        ListFree( lines );
        ListFree( points );
        return ok;
    }

    i32 WorldRemapPrimitive( i32 held, i32 removed ) {
        if( held == removed ) {
            return kNoPrimitive;
        }
        if( held > removed ) {
            return held - 1;
        }
        return held;
    }

    bool WorldRemovePrimitive( World & world, Renderer * r, i32 primitive ) {
        if( primitive < 0 || primitive >= world.primitives.count ) {
            return false;
        }

        // Cleared while the mesh is still alive, so the highlight is dropped
        // properly rather than left on whatever slides into this index.
        if( world.selected == primitive ) {
            WorldSetSelected( world, r, kNoPrimitive );
        }

        Primitive & entry = world.primitives[primitive];
        // Idles the device, so the buffers are not pulled out from under a
        // frame that is still reading them.
        RendererDestroyStaticMesh( r, entry.renderMesh );
        HalfMeshFree( entry.halfMesh );

        ListRemoveIndex( world.primitives, primitive );

        // A selection above the hole has just shifted down with everything
        // else, so the index it holds now names its old neighbour.
        world.selected = WorldRemapPrimitive( world.selected, primitive );
        return true;
    }

    void WorldFree( World & world ) {
        for( i32 i = 0; i < world.primitives.count; i++ ) {
            HalfMeshFree( world.primitives[i].halfMesh );
        }
        ListFree( world.primitives );
    }

    bool WorldCreateDefaultLevel( World & world, Renderer * r ) {
        RenderMaterial material = RenderMaterialDefault();

        // A cube is centred on its own origin, so lifting it by half its height
        // stands it on the grid rather than half sunk through it.
        HalfMesh cube = {};
        HalfMeshCreateCube( cube, 2 );
        material.albedo = Vec3{ 0.72f, 0.74f, 0.78f };
        Transform cubeTransform = TransformDefault();
        cubeTransform.position = Vec3{ -1.6f, 1.0f, 0.0f };
        bool ok = WorldAddPrimitive( world, r, cube, material, cubeTransform ) != kNoPrimitive;
        HalfMeshFree( cube );

        HalfMesh tower = {};
        HalfMeshCreateCube( tower, 2 );

        // Found rather than assumed, so a change to the cube's face order does
        // not quietly extrude a wall sideways.
        i32 topFace = kHMNone;
        for( i32 f = 0; f < tower.faces.count; f++ ) {
            if( tower.faces[f].normal.y > 0.9f ) {
                topFace = f;
            }
        }
        if( topFace != kHMNone ) {
            HalfMeshExtrudeFace( tower, topFace, 1.5f );
        }

        material.albedo = Vec3{ 0.78f, 0.62f, 0.45f };
        Transform towerTransform = TransformDefault();
        towerTransform.position = Vec3{ 1.6f, 1.0f, 0.0f };
        ok = ok && WorldAddPrimitive( world, r, tower, material, towerTransform ) != kNoPrimitive;
        HalfMeshFree( tower );

        if( !ok ) {
            fprintf( stderr, "Failed to build the default level\n" );
        }
        return ok;
    }

} // namespace sol
