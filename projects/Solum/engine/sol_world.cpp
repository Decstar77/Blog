#include "sol_world.h"

#include <cstdio>

namespace sol {

    bool RenderMeshFromHalfMesh( Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material, RenderStaticMesh * outMesh ) {
        List<HMTriVertex> triVertices = {};
        List<u32> triIndices = {};
        HalfMeshTriangulate( halfMesh, triVertices, triIndices );

        List<StaticMeshVertex> vertices = {};
        bool ok = triVertices.count > 0 && triIndices.count > 0;

        if( ok ) {
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

            ok = RenderStaticMeshCreate( r, vertices.data, vertices.count, triIndices.data, triIndices.count, material.texture, outMesh );
        }

        ListFree( triVertices );
        ListFree( triIndices );
        ListFree( vertices );
        return ok;
    }

    i32 WorldAddPrimitive( World & world, Renderer * r, const HalfMesh & halfMesh, const RenderMaterial & material, const Mat4 & transform ) {
        RenderStaticMesh renderMesh = {};
        if( !RenderMeshFromHalfMesh( r, halfMesh, material, &renderMesh ) ) {
            return kNoPrimitive;
        }
        renderMesh.transform = transform;

        if( RendererAddStaticMesh( r, renderMesh ) == nullptr ) {
            RenderStaticMeshDestroy( r, &renderMesh );
            return kNoPrimitive;
        }

        Primitive primitive = {};
        primitive.halfMesh = HalfMeshCopy( halfMesh );
        primitive.material = material;
        // RendererAddStaticMesh appends, so the slot it landed in is the last.
        primitive.renderMesh = r->staticMeshes.count - 1;

        if( ListAdd( world.primitives, primitive ) == nullptr ) {
            HalfMeshFree( primitive.halfMesh );
            return kNoPrimitive;
        }

        return world.primitives.count - 1;
    }

    bool WorldRebuildPrimitive( World & world, Renderer * r, i32 primitive ) {
        if( primitive < 0 || primitive >= world.primitives.count ) {
            return false;
        }

        Primitive & entry = world.primitives[primitive];
        if( entry.renderMesh < 0 || entry.renderMesh >= r->staticMeshes.count ) {
            return false;
        }

        RenderStaticMesh rebuilt = {};
        if( !RenderMeshFromHalfMesh( r, entry.halfMesh, entry.material, &rebuilt ) ) {
            return false;
        }

        // Built the replacement first, so a failure above leaves the old mesh
        // on screen rather than a hole.
        RenderStaticMesh * slot = &r->staticMeshes[entry.renderMesh];
        rebuilt.transform = slot->transform;

        // The buffers about to be freed may still be referenced by a frame the
        // GPU has not finished with.
        vkDeviceWaitIdle( r->device );
        RenderStaticMeshDestroy( r, slot );
        *slot = rebuilt;
        return true;
    }

    void WorldSetPrimitiveTransform( World & world, Renderer * r, i32 primitive, const Mat4 & transform ) {
        if( primitive < 0 || primitive >= world.primitives.count ) {
            return;
        }

        const i32 renderMesh = world.primitives[primitive].renderMesh;
        if( renderMesh < 0 || renderMesh >= r->staticMeshes.count ) {
            return;
        }

        // Read fresh on the CPU when the next frame records, so there is
        // nothing to synchronise against here.
        r->staticMeshes[renderMesh].transform = transform;
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
