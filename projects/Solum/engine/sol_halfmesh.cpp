#include "sol_halfmesh.h"

#include <cmath>

namespace sol {

    void HalfMeshFree( HalfMesh & mesh ) {
        ListFree( mesh.vertices );
        ListFree( mesh.edges );
        ListFree( mesh.faces );
        ListFree( mesh.halfEdges );
    }

    void HalfMeshClear( HalfMesh & mesh ) {
        ListClear( mesh.vertices );
        ListClear( mesh.edges );
        ListClear( mesh.faces );
        ListClear( mesh.halfEdges );
    }

    HalfMesh HalfMeshCopy( const HalfMesh & mesh ) {
        HalfMesh result = {};
        result.vertices = ListCopy( mesh.vertices );
        result.edges = ListCopy( mesh.edges );
        result.faces = ListCopy( mesh.faces );
        result.halfEdges = ListCopy( mesh.halfEdges );
        return result;
    }

    // --- construction -------------------------------------------------------

    bool HalfMeshFromPolygons( HalfMesh & mesh, const Vec3 * positions, i32 positionCount,
                               const i32 * faceIndices, const i32 * faceSizes, i32 faceCount ) {
        HalfMeshClear( mesh );

        if( positionCount <= 0 || faceCount <= 0 ) {
            return false;
        }

        for( i32 i = 0; i < positionCount; i++ ) {
            HMVertex vertex = {};
            vertex.position = positions[i];
            vertex.halfEdge = kHMNone;
            ListAdd( mesh.vertices, vertex );
        }

        // Pass one: a half-edge per corner, linked into its face loop. Twins and
        // edges come later, once every corner exists to be matched against.
        i32 cursor = 0;
        for( i32 f = 0; f < faceCount; f++ ) {
            const i32 sideCount = faceSizes[f];
            if( sideCount < 3 ) {
                HalfMeshClear( mesh );
                return false;
            }

            const i32 base = mesh.halfEdges.count;
            for( i32 i = 0; i < sideCount; i++ ) {
                const i32 vertex = faceIndices[cursor + i];
                if( vertex < 0 || vertex >= positionCount ) {
                    HalfMeshClear( mesh );
                    return false;
                }

                // A face that visits a vertex twice pinches itself, and no
                // consistent twin assignment exists for it.
                for( i32 j = 0; j < i; j++ ) {
                    if( faceIndices[cursor + j] == vertex ) {
                        HalfMeshClear( mesh );
                        return false;
                    }
                }

                HalfEdge halfEdge = {};
                halfEdge.next = base + ( i + 1 ) % sideCount;
                halfEdge.twin = kHMNone;
                halfEdge.vert = vertex;
                halfEdge.edge = kHMNone;
                halfEdge.face = f;
                ListAdd( mesh.halfEdges, halfEdge );
            }

            HMFace face = {};
            face.halfEdge = base;
            ListAdd( mesh.faces, face );

            cursor += sideCount;
        }

        const i32 interiorCount = mesh.halfEdges.count;

        // Pass two: bucket half-edges by origin with a counting sort, so finding
        // the twin of (a, b) only scans the handful of half-edges leaving b
        // rather than the whole mesh.
        List<i32> bucketStart = {};
        ListResize( bucketStart, positionCount + 1 );
        for( i32 h = 0; h < interiorCount; h++ ) {
            bucketStart[mesh.halfEdges[h].vert + 1]++;
        }
        for( i32 v = 0; v < positionCount; v++ ) {
            bucketStart[v + 1] += bucketStart[v];
        }

        List<i32> bucketFill = ListCopy( bucketStart );
        List<i32> buckets = {};
        ListResize( buckets, interiorCount );
        for( i32 h = 0; h < interiorCount; h++ ) {
            buckets[bucketFill[mesh.halfEdges[h].vert]++] = h;
        }
        ListFree( bucketFill );

        bool ok = true;

        // No two half-edges may run the same way along one edge. That happens
        // when three or more faces share the edge, or when two share it with
        // matching winding, and it has to be caught here: the twin search below
        // only ever looks for a REVERSED half-edge, so same-direction duplicates
        // would sail through it and be mistaken for boundary edges.
        for( i32 h = 0; h < interiorCount && ok; h++ ) {
            const i32 origin = mesh.halfEdges[h].vert;
            const i32 dest = mesh.halfEdges[mesh.halfEdges[h].next].vert;

            for( i32 i = bucketStart[origin]; i < bucketStart[origin + 1]; i++ ) {
                const i32 other = buckets[i];
                if( other != h && mesh.halfEdges[mesh.halfEdges[other].next].vert == dest ) {
                    ok = false;
                    break;
                }
            }
        }

        for( i32 h = 0; h < interiorCount && ok; h++ ) {
            if( mesh.halfEdges[h].twin != kHMNone ) {
                continue;
            }

            const i32 origin = mesh.halfEdges[h].vert;
            // Twins are not set yet, so the destination has to come from next.
            const i32 dest = mesh.halfEdges[mesh.halfEdges[h].next].vert;

            i32 twin = kHMNone;
            for( i32 i = bucketStart[dest]; i < bucketStart[dest + 1]; i++ ) {
                const i32 candidate = buckets[i];
                const i32 candidateDest = mesh.halfEdges[mesh.halfEdges[candidate].next].vert;
                if( candidateDest != origin ) {
                    continue;
                }

                // A second face already claimed this side, so three faces meet
                // along one edge and the mesh is not manifold.
                if( mesh.halfEdges[candidate].twin != kHMNone ) {
                    ok = false;
                }
                twin = candidate;
                break;
            }

            if( !ok ) {
                break;
            }

            HMEdge edge = {};
            edge.halfEdge = h;
            const i32 edgeIndex = mesh.edges.count;
            ListAdd( mesh.edges, edge );

            mesh.halfEdges[h].edge = edgeIndex;
            if( twin != kHMNone ) {
                mesh.halfEdges[h].twin = twin;
                mesh.halfEdges[twin].twin = h;
                mesh.halfEdges[twin].edge = edgeIndex;
            }
        }

        ListFree( bucketStart );
        ListFree( buckets );

        if( !ok ) {
            HalfMeshClear( mesh );
            return false;
        }

        // Pass three: every unmatched half-edge borders a hole, so give it a
        // twin on the far side. After this twin is never kHMNone.
        List<i32> boundaryOutgoing = {};
        ListResize( boundaryOutgoing, positionCount );
        ListFill( boundaryOutgoing, kHMNone );

        for( i32 h = 0; h < interiorCount; h++ ) {
            if( mesh.halfEdges[h].twin != kHMNone ) {
                continue;
            }

            const i32 dest = mesh.halfEdges[mesh.halfEdges[h].next].vert;

            HalfEdge boundary = {};
            boundary.next = kHMNone;    // linked below, once they all exist
            boundary.twin = h;
            boundary.vert = dest;
            boundary.edge = mesh.halfEdges[h].edge;
            boundary.face = kHMNone;

            const i32 boundaryIndex = mesh.halfEdges.count;
            ListAdd( mesh.halfEdges, boundary );
            mesh.halfEdges[h].twin = boundaryIndex;

            // A manifold vertex sits on at most one hole, so one slot is enough
            // - and a second claim on that slot is itself the bug. Two holes
            // meeting at a single vertex is a bowtie, and overwriting here
            // would strand the earlier half-edge outside any boundary loop.
            if( boundaryOutgoing[dest] != kHMNone ) {
                ok = false;
            }
            boundaryOutgoing[dest] = boundaryIndex;
        }

        // A boundary loop runs the opposite way round to the face loops beside
        // it, so the next boundary half-edge is the one leaving this one's
        // destination - which is the origin of its interior twin.
        for( i32 h = interiorCount; h < mesh.halfEdges.count; h++ ) {
            const i32 interior = mesh.halfEdges[h].twin;
            const i32 dest = mesh.halfEdges[interior].vert;
            mesh.halfEdges[h].next = boundaryOutgoing[dest];
            if( mesh.halfEdges[h].next == kHMNone ) {
                ok = false;
            }
        }

        // Prefer a boundary half-edge as the vertex's representative, so a
        // rotation starting there reaches the hole on the first step.
        for( i32 h = 0; h < mesh.halfEdges.count; h++ ) {
            const i32 origin = mesh.halfEdges[h].vert;
            if( mesh.vertices[origin].halfEdge == kHMNone ||
                mesh.halfEdges[h].face == kHMNone ) {
                mesh.vertices[origin].halfEdge = h;
            }
        }

        ListFree( boundaryOutgoing );

        if( !ok ) {
            HalfMeshClear( mesh );
            return false;
        }

        HalfMeshComputeFaceNormals( mesh );
        return true;
    }

    void HalfMeshToPolygons( const HalfMesh & mesh, List<Vec3> & outPositions,
                             List<i32> & outFaceIndices, List<i32> & outFaceSizes ) {
        ListClear( outPositions );
        ListClear( outFaceIndices );
        ListClear( outFaceSizes );

        for( i32 v = 0; v < mesh.vertices.count; v++ ) {
            ListAdd( outPositions, mesh.vertices[v].position );
        }

        for( i32 f = 0; f < mesh.faces.count; f++ ) {
            const i32 start = mesh.faces[f].halfEdge;
            i32 sideCount = 0;
            i32 h = start;
            do {
                ListAdd( outFaceIndices, mesh.halfEdges[h].vert );
                sideCount++;
                h = mesh.halfEdges[h].next;
            } while( h != start );

            ListAdd( outFaceSizes, sideCount );
        }
    }

    void HalfMeshCreateTriangle( HalfMesh & mesh ) {
        const Vec3 positions[] = {
            { -0.5f, -0.5f, 0.0f },
            {  0.5f, -0.5f, 0.0f },
            {  0.0f,  0.5f, 0.0f },
        };
        const i32 indices[] = { 0, 1, 2 };
        const i32 sizes[] = { 3 };

        HalfMeshFromPolygons( mesh, positions, (i32)SPLATS_ARRAY_COUNT( positions ),
                              indices, sizes, 1 );
    }

    void HalfMeshCreateQuad( HalfMesh & mesh, f32 size ) {
        const f32 half = size * 0.5f;
        const Vec3 positions[] = {
            { -half, 0.0f, -half },
            { -half, 0.0f,  half },
            {  half, 0.0f,  half },
            {  half, 0.0f, -half },
        };
        const i32 indices[] = { 0, 1, 2, 3 };
        const i32 sizes[] = { 4 };

        HalfMeshFromPolygons( mesh, positions, (i32)SPLATS_ARRAY_COUNT( positions ),
                              indices, sizes, 1 );
    }

    void HalfMeshCreateCube( HalfMesh & mesh, i32 size ) {
        const f32 half = (f32)size * 0.5f;
        const Vec3 positions[] = {
            { -half, -half, -half },    // 0
            {  half, -half, -half },    // 1
            {  half,  half, -half },    // 2
            { -half,  half, -half },    // 3
            { -half, -half,  half },    // 4
            {  half, -half,  half },    // 5
            {  half,  half,  half },    // 6
            { -half,  half,  half },    // 7
        };

        // Each quad wound counter-clockwise seen from outside, so every face
        // normal points away from the centre.
        const i32 indices[] = {
            4, 5, 6, 7,     // +z
            1, 0, 3, 2,     // -z
            0, 4, 7, 3,     // -x
            5, 1, 2, 6,     // +x
            3, 7, 6, 2,     // +y
            0, 1, 5, 4,     // -y
        };
        const i32 sizes[] = { 4, 4, 4, 4, 4, 4 };

        HalfMeshFromPolygons( mesh, positions, (i32)SPLATS_ARRAY_COUNT( positions ),
                              indices, sizes, (i32)SPLATS_ARRAY_COUNT( sizes ) );
    }

    void HalfMeshCreatePlane( HalfMesh & mesh, i32 cols, i32 rows, f32 cellSize ) {
        if( cols < 1 || rows < 1 ) {
            HalfMeshClear( mesh );
            return;
        }

        // Centred on the origin, so a plane drops in around whatever is already
        // at the middle of the grid.
        const f32 originX = -0.5f * (f32)cols * cellSize;
        const f32 originZ = -0.5f * (f32)rows * cellSize;

        List<Vec3> positions = {};
        for( i32 z = 0; z <= rows; z++ ) {
            for( i32 x = 0; x <= cols; x++ ) {
                ListAdd( positions, Vec3{ originX + (f32)x * cellSize, 0.0f,
                                          originZ + (f32)z * cellSize } );
            }
        }

        const i32 stride = cols + 1;
        List<i32> indices = {};
        List<i32> sizes = {};
        for( i32 z = 0; z < rows; z++ ) {
            for( i32 x = 0; x < cols; x++ ) {
                // Wound so the normal comes out +y rather than into the floor.
                ListAdd( indices, z * stride + x );
                ListAdd( indices, ( z + 1 ) * stride + x );
                ListAdd( indices, ( z + 1 ) * stride + x + 1 );
                ListAdd( indices, z * stride + x + 1 );
                ListAdd( sizes, 4 );
            }
        }

        HalfMeshFromPolygons( mesh, positions.data, positions.count,
                              indices.data, sizes.data, sizes.count );

        ListFree( positions );
        ListFree( indices );
        ListFree( sizes );
    }

    // --- traversal ----------------------------------------------------------

    i32 HalfEdgePrev( const HalfMesh & mesh, i32 halfEdge ) {
        i32 h = halfEdge;
        while( mesh.halfEdges[h].next != halfEdge ) {
            h = mesh.halfEdges[h].next;
        }
        return h;
    }

    i32 HalfMeshFaceVertexCount( const HalfMesh & mesh, i32 face ) {
        const i32 start = mesh.faces[face].halfEdge;
        i32 count = 0;
        i32 h = start;
        do {
            count++;
            h = mesh.halfEdges[h].next;
        } while( h != start );
        return count;
    }

    void HalfMeshFaceHalfEdges( const HalfMesh & mesh, i32 face, List<i32> & outHalfEdges ) {
        ListClear( outHalfEdges );

        const i32 start = mesh.faces[face].halfEdge;
        i32 h = start;
        do {
            ListAdd( outHalfEdges, h );
            h = mesh.halfEdges[h].next;
        } while( h != start );
    }

    void HalfMeshFaceVertices( const HalfMesh & mesh, i32 face, List<i32> & outVertices ) {
        ListClear( outVertices );

        const i32 start = mesh.faces[face].halfEdge;
        i32 h = start;
        do {
            ListAdd( outVertices, mesh.halfEdges[h].vert );
            h = mesh.halfEdges[h].next;
        } while( h != start );
    }

    void HalfMeshVertexOutgoing( const HalfMesh & mesh, i32 vertex, List<i32> & outHalfEdges ) {
        ListClear( outHalfEdges );

        const i32 start = mesh.vertices[vertex].halfEdge;
        if( start == kHMNone ) {
            return;
        }

        i32 h = start;
        do {
            ListAdd( outHalfEdges, h );
            h = HalfEdgeRotateAroundVertex( mesh, h );
        } while( h != start );
    }

    i32 HalfMeshVertexValence( const HalfMesh & mesh, i32 vertex ) {
        const i32 start = mesh.vertices[vertex].halfEdge;
        if( start == kHMNone ) {
            return 0;
        }

        i32 count = 0;
        i32 h = start;
        do {
            count++;
            h = HalfEdgeRotateAroundVertex( mesh, h );
        } while( h != start );
        return count;
    }

    Vec3 HalfMeshFaceCentroid( const HalfMesh & mesh, i32 face ) {
        const i32 start = mesh.faces[face].halfEdge;
        Vec3 sum = {};
        i32 count = 0;

        i32 h = start;
        do {
            sum = sum + mesh.vertices[mesh.halfEdges[h].vert].position;
            count++;
            h = mesh.halfEdges[h].next;
        } while( h != start );

        if( count == 0 ) {
            return sum;
        }
        return sum * ( 1.0f / (f32)count );
    }

    Vec3 HalfMeshFaceNormal( const HalfMesh & mesh, i32 face ) {
        // Newell's method: sums the signed areas the loop projects onto each
        // axis plane. Every corner contributes, so a warped quad gets the
        // best-fit normal instead of one read off three arbitrary corners.
        const i32 start = mesh.faces[face].halfEdge;
        Vec3 normal = {};

        i32 h = start;
        do {
            const Vec3 current = mesh.vertices[mesh.halfEdges[h].vert].position;
            const i32 nextHalfEdge = mesh.halfEdges[h].next;
            const Vec3 next = mesh.vertices[mesh.halfEdges[nextHalfEdge].vert].position;

            normal.x += ( current.y - next.y ) * ( current.z + next.z );
            normal.y += ( current.z - next.z ) * ( current.x + next.x );
            normal.z += ( current.x - next.x ) * ( current.y + next.y );

            h = nextHalfEdge;
        } while( h != start );

        return Vec3Normalize( normal );
    }

    void HalfMeshComputeFaceNormals( HalfMesh & mesh ) {
        for( i32 f = 0; f < mesh.faces.count; f++ ) {
            mesh.faces[f].normal = HalfMeshFaceNormal( mesh, f );
        }
    }

    Vec3 HalfMeshEdgeMidpoint( const HalfMesh & mesh, i32 edge ) {
        const i32 h = mesh.edges[edge].halfEdge;
        const Vec3 a = mesh.vertices[mesh.halfEdges[h].vert].position;
        const Vec3 b = mesh.vertices[HalfEdgeDest( mesh, h )].position;
        return ( a + b ) * 0.5f;
    }

    // --- editing ------------------------------------------------------------

    i32 HalfMeshExtrudeFace( HalfMesh & mesh, i32 face, f32 distance ) {
        if( face < 0 || face >= mesh.faces.count ) {
            return kHMNone;
        }

        List<i32> loop = {};
        HalfMeshFaceVertices( mesh, face, loop );
        const i32 sideCount = loop.count;

        const Vec3 offset = mesh.faces[face].normal * distance;

        List<Vec3> positions = {};
        List<i32> indices = {};
        List<i32> sizes = {};
        HalfMeshToPolygons( mesh, positions, indices, sizes );

        // The lifted ring goes on the end, which is what keeps the existing
        // vertex indices meaning what they meant before the edit.
        const i32 firstNewVertex = positions.count;
        for( i32 i = 0; i < sideCount; i++ ) {
            ListAdd( positions, mesh.vertices[loop[i]].position + offset );
        }

        // Rewrite the extruded face in place as the cap. Its slot in the index
        // stream is the same width, since the cap has as many sides as before.
        i32 cursor = 0;
        for( i32 f = 0; f < face; f++ ) {
            cursor += sizes[f];
        }
        for( i32 i = 0; i < sideCount; i++ ) {
            indices[cursor + i] = firstNewVertex + i;
        }

        // One quad per original edge, wound so it faces outward: up the far
        // side, across the top, back down the near side.
        for( i32 i = 0; i < sideCount; i++ ) {
            const i32 next = ( i + 1 ) % sideCount;
            ListAdd( indices, loop[i] );
            ListAdd( indices, loop[next] );
            ListAdd( indices, firstNewVertex + next );
            ListAdd( indices, firstNewVertex + i );
            ListAdd( sizes, 4 );
        }

        const bool ok = HalfMeshFromPolygons( mesh, positions.data, positions.count,
                                              indices.data, sizes.data, sizes.count );

        ListFree( loop );
        ListFree( positions );
        ListFree( indices );
        ListFree( sizes );

        return ok ? face : kHMNone;
    }

    i32 HalfMeshSplitEdge( HalfMesh & mesh, i32 edge, f32 t ) {
        if( edge < 0 || edge >= mesh.edges.count ) {
            return kHMNone;
        }

        const i32 h = mesh.edges[edge].halfEdge;
        const i32 vertexA = mesh.halfEdges[h].vert;
        const i32 vertexB = HalfEdgeDest( mesh, h );

        const Vec3 a = mesh.vertices[vertexA].position;
        const Vec3 b = mesh.vertices[vertexB].position;

        List<Vec3> positions = {};
        List<i32> indices = {};
        List<i32> sizes = {};
        HalfMeshToPolygons( mesh, positions, indices, sizes );

        const i32 newVertex = positions.count;
        ListAdd( positions, a + ( b - a ) * t );

        // Walk each face looking for the two corners in sequence. Whichever way
        // round the face runs, the new vertex belongs between them.
        List<i32> rebuilt = {};
        i32 cursor = 0;
        for( i32 f = 0; f < sizes.count; f++ ) {
            const i32 sideCount = sizes[f];
            i32 inserted = 0;

            for( i32 i = 0; i < sideCount; i++ ) {
                const i32 current = indices[cursor + i];
                const i32 next = indices[cursor + ( i + 1 ) % sideCount];

                ListAdd( rebuilt, current );
                const bool matches = ( current == vertexA && next == vertexB ) ||
                                     ( current == vertexB && next == vertexA );
                if( matches ) {
                    ListAdd( rebuilt, newVertex );
                    inserted++;
                }
            }

            sizes[f] = sideCount + inserted;
            cursor += sideCount;
        }

        const bool ok = HalfMeshFromPolygons( mesh, positions.data, positions.count,
                                              rebuilt.data, sizes.data, sizes.count );

        ListFree( positions );
        ListFree( indices );
        ListFree( sizes );
        ListFree( rebuilt );

        return ok ? newVertex : kHMNone;
    }

    // --- output -------------------------------------------------------------

    static Vec2 PlanarUv( Vec3 position, Vec3 normal ) {
        const f32 absX = normal.x < 0.0f ? -normal.x : normal.x;
        const f32 absY = normal.y < 0.0f ? -normal.y : normal.y;
        const f32 absZ = normal.z < 0.0f ? -normal.z : normal.z;

        // Project along whichever axis the face most faces, so a wall is not
        // squashed into a sliver of the texture.
        if( absY >= absX && absY >= absZ ) {
            return Vec2{ position.x, position.z };
        }
        if( absX >= absZ ) {
            return Vec2{ position.z, position.y };
        }
        return Vec2{ position.x, position.y };
    }

    void HalfMeshTriangulate( const HalfMesh & mesh, List<HMTriVertex> & outVertices,
                              List<u32> & outIndices ) {
        ListClear( outVertices );
        ListClear( outIndices );

        for( i32 f = 0; f < mesh.faces.count; f++ ) {
            const Vec3 normal = mesh.faces[f].normal;
            const i32 base = outVertices.count;

            const i32 start = mesh.faces[f].halfEdge;
            i32 sideCount = 0;
            i32 h = start;
            do {
                const Vec3 position = mesh.vertices[mesh.halfEdges[h].vert].position;

                HMTriVertex vertex = {};
                vertex.position = position;
                vertex.normal = normal;
                vertex.uv = PlanarUv( position, normal );
                ListAdd( outVertices, vertex );

                sideCount++;
                h = mesh.halfEdges[h].next;
            } while( h != start );

            // Fan from the first corner. Fine for the convex faces a level
            // builder produces; a concave n-gon would need ear clipping.
            for( i32 i = 1; i + 1 < sideCount; i++ ) {
                ListAdd( outIndices, (u32)base );
                ListAdd( outIndices, (u32)( base + i ) );
                ListAdd( outIndices, (u32)( base + i + 1 ) );
            }
        }
    }

    // --- debugging ----------------------------------------------------------

    bool HalfMeshValidate( const HalfMesh & mesh, const char ** outError ) {
        const char * error = nullptr;

        for( i32 h = 0; h < mesh.halfEdges.count && error == nullptr; h++ ) {
            const HalfEdge & halfEdge = mesh.halfEdges[h];

            if( halfEdge.twin < 0 || halfEdge.twin >= mesh.halfEdges.count ) {
                error = "half-edge has no twin";
            } else if( mesh.halfEdges[halfEdge.twin].twin != h ) {
                error = "twin is not mutual";
            } else if( halfEdge.twin == h ) {
                error = "half-edge is its own twin";
            } else if( halfEdge.next < 0 || halfEdge.next >= mesh.halfEdges.count ) {
                error = "half-edge has no next";
            } else if( halfEdge.vert < 0 || halfEdge.vert >= mesh.vertices.count ) {
                error = "half-edge has no origin vertex";
            } else if( halfEdge.edge < 0 || halfEdge.edge >= mesh.edges.count ) {
                error = "half-edge has no edge";
            } else if( mesh.halfEdges[halfEdge.twin].edge != halfEdge.edge ) {
                error = "twins disagree about their edge";
            } else if( halfEdge.face != kHMNone &&
                       ( halfEdge.face < 0 || halfEdge.face >= mesh.faces.count ) ) {
                error = "half-edge has an out of range face";
            } else if( mesh.halfEdges[halfEdge.next].face != halfEdge.face ) {
                error = "a loop crosses between two faces";
            } else if( mesh.halfEdges[halfEdge.next].vert != HalfEdgeDest( mesh, h ) ) {
                error = "next does not start where this half-edge ends";
            }
        }

        // Every loop has to come back to where it started, and within the number
        // of steps the mesh actually has - otherwise a broken next chain would
        // spin here forever.
        for( i32 f = 0; f < mesh.faces.count && error == nullptr; f++ ) {
            const i32 start = mesh.faces[f].halfEdge;
            if( start < 0 || start >= mesh.halfEdges.count ) {
                error = "face has no half-edge";
                break;
            }

            i32 steps = 0;
            i32 h = start;
            do {
                if( mesh.halfEdges[h].face != f ) {
                    error = "face loop visits a half-edge belonging to another face";
                    break;
                }
                h = mesh.halfEdges[h].next;
                steps++;
            } while( h != start && steps <= mesh.halfEdges.count );

            if( error == nullptr && h != start ) {
                error = "face loop does not close";
            }
        }

        for( i32 e = 0; e < mesh.edges.count && error == nullptr; e++ ) {
            const i32 h = mesh.edges[e].halfEdge;
            if( h < 0 || h >= mesh.halfEdges.count ) {
                error = "edge has no half-edge";
            } else if( mesh.halfEdges[h].edge != e ) {
                error = "edge and half-edge disagree";
            }
        }

        for( i32 v = 0; v < mesh.vertices.count && error == nullptr; v++ ) {
            const i32 h = mesh.vertices[v].halfEdge;
            if( h == kHMNone ) {
                continue;   // an unused vertex is allowed, just not referenced
            }
            if( h < 0 || h >= mesh.halfEdges.count ) {
                error = "vertex has an out of range half-edge";
            } else if( mesh.halfEdges[h].vert != v ) {
                error = "vertex half-edge does not start at that vertex";
            }
        }

        if( outError != nullptr ) {
            *outError = error;
        }
        return error == nullptr;
    }

} // namespace sol
