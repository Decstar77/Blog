#include "sol_world.h"

#include <cstdio>

namespace sol {

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

    i32 WorldAddPrimitive( World & world, Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material, const Mat4 & transform ) {
        RenderMeshHandle renderMesh = RenderMeshFromHalfMesh( r, halfMesh, material );
        RenderStaticMesh * mesh = RendererGetStaticMesh( r, renderMesh );
        if( mesh == nullptr ) {
            return kNoPrimitive;
        }
        mesh->transform = transform;

        Primitive primitive = {};
        primitive.halfMesh = HalfMeshCopy( halfMesh );
        primitive.material = material;
        primitive.renderMesh = renderMesh;

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

        // Carried across the rebuild, or a selected primitive would drop its
        // highlight while it stayed selected.
        const Mat4 transform = previous->transform;
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

    void WorldSetPrimitiveTransform( World & world, Renderer * r, i32 primitive, const Mat4 & transform ) {
        if( primitive < 0 || primitive >= world.primitives.count ) {
            return;
        }

        RenderStaticMesh * mesh = RendererGetStaticMesh( r, world.primitives[primitive].renderMesh );
        if( mesh == nullptr ) {
            return;
        }

        // Read fresh on the CPU when the next frame records, so there is
        // nothing to synchronise against here.
        mesh->transform = transform;
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
        bool ok = WorldAddPrimitive( world, r, cube, material,
                                     Mat4Translate( Vec3{ -1.6f, 1.0f, 0.0f } ) ) != kNoPrimitive;
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
        ok = ok && WorldAddPrimitive( world, r, tower, material, Mat4Translate( Vec3{ 1.6f, 1.0f, 0.0f } ) ) != kNoPrimitive;
        HalfMeshFree( tower );

        if( !ok ) {
            fprintf( stderr, "Failed to build the default level\n" );
        }
        return ok;
    }

} // namespace sol
