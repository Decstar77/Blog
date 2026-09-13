#pragma once
#include "sol_defines.h"
#include "sol_list.h"
#include "sol_math.h"

namespace sol {

    // An index into one of HalfMesh's four lists. kHMNone is "no such element":
    // the face of a boundary half-edge, or a slot not filled in yet.
    constexpr i32 kHMNone = -1;

    struct HMVertex {
        Vec3    position;
        // Any one half-edge with this vertex as its origin. Boundary vertices
        // store their boundary half-edge, so a walk that starts here meets the
        // hole immediately instead of after a full lap.
        i32     halfEdge;
    };

    struct HMEdge {
        // One of the pair. The other is halfEdges[halfEdge].twin.
        i32     halfEdge;
    };

    struct HMFace {
        i32     halfEdge;
        // Only valid after HalfMeshComputeFaceNormals; the builders call it.
        Vec3    normal;
    };

    struct HalfEdge {
        i32     next;
        i32     twin;
        i32     vert;   // vertex this half-edge points FROM (origin)
        i32     edge;
        i32     face;   // kHMNone for boundary half-edges
    };

    // A half-edge mesh over arbitrary polygons - it is not limited to triangles,
    // which is the point of using one for a level builder: a wall stays one quad
    // face until something actually splits it.
    //
    // The invariants every function here maintains, and that HalfMeshValidate
    // checks:
    //   - every half-edge has a twin, and twin(twin(h)) == h
    //   - a hole is not an absence of half-edges but a loop of them with
    //     face == kHMNone, so twin is never kHMNone and needs no null check
    //   - next walks a face loop counter-clockwise seen from the face's front;
    //     a boundary loop therefore walks clockwise around its hole
    //   - halfEdges[h].edge is shared by h and its twin
    struct HalfMesh {
        List<HMVertex>  vertices;
        List<HMEdge>    edges;
        List<HMFace>    faces;
        List<HalfEdge>  halfEdges;
    };

    // Flat-shaded triangle output. Faces do not share vertices here, because a
    // level's corners want a crease rather than an averaged normal.
    struct HMTriVertex {
        Vec3    position;
        Vec3    normal;
        Vec2    uv;
    };

    // --- lifetime -----------------------------------------------------------

    void        HalfMeshFree( HalfMesh & mesh );
    // Keeps the allocations and drops the contents, for rebuilding in place.
    void        HalfMeshClear( HalfMesh & mesh );
    HalfMesh    HalfMeshCopy( const HalfMesh & mesh );

    // --- construction -------------------------------------------------------

    // The one real constructor: an indexed polygon soup in, connectivity out.
    // faceIndices is every face's vertices back to back, faceSizes says how many
    // each face takes. Winding decides which way a face points, so wind them
    // counter-clockwise seen from the front.
    //
    // Returns false and leaves the mesh empty if the soup is malformed or not
    // manifold - more than two faces meeting at one edge, or a face reusing a
    // vertex. Failing loudly matters here: half-built connectivity is far worse
    // to debug later than a rejected input now.
    bool HalfMeshFromPolygons( HalfMesh & mesh, const Vec3 * positions, i32 positionCount,
                               const i32 * faceIndices, const i32 * faceSizes, i32 faceCount );

    // The inverse. Edit operations go out to this representation, change it, and
    // rebuild - see the note on HalfMeshExtrudeFace.
    void HalfMeshToPolygons( const HalfMesh & mesh, List<Vec3> & outPositions,
                             List<i32> & outFaceIndices, List<i32> & outFaceSizes );

    // Primitives. Each is one polygon soup handed to HalfMeshFromPolygons, so
    // they exercise the same path everything else does.
    void HalfMeshCreateTriangle( HalfMesh & mesh );
    void HalfMeshCreateQuad( HalfMesh & mesh, f32 size );
    // Edge length in grid units, centred on the origin.
    void HalfMeshCreateCube( HalfMesh & mesh, i32 size );
    // A flat grid of quads in the xz plane, facing +y. The level builder's floor.
    void HalfMeshCreatePlane( HalfMesh & mesh, i32 cols, i32 rows, f32 cellSize );

    // --- traversal ----------------------------------------------------------

    // Destination of a half-edge: the origin of its twin. Read through the twin
    // rather than through next, because next is the half that an edit breaks
    // first.
    inline i32 HalfEdgeDest( const HalfMesh & mesh, i32 halfEdge ) {
        return mesh.halfEdges[mesh.halfEdges[halfEdge].twin].vert;
    }

    inline bool HalfEdgeIsBoundary( const HalfMesh & mesh, i32 halfEdge ) {
        return mesh.halfEdges[halfEdge].face == kHMNone;
    }

    // An edge is on the boundary when either side of it is.
    inline bool HalfMeshEdgeIsBoundary( const HalfMesh & mesh, i32 edge ) {
        const i32 h = mesh.edges[edge].halfEdge;
        return HalfEdgeIsBoundary( mesh, h ) || HalfEdgeIsBoundary( mesh, mesh.halfEdges[h].twin );
    }

    // O(loop): there are no prev links, so this laps the face.
    i32     HalfEdgePrev( const HalfMesh & mesh, i32 halfEdge );
    // Next half-edge leaving the same vertex, rotating one step. Lands on the
    // boundary loop rather than stopping, so a full rotation always terminates.
    inline i32 HalfEdgeRotateAroundVertex( const HalfMesh & mesh, i32 halfEdge ) {
        return mesh.halfEdges[mesh.halfEdges[halfEdge].twin].next;
    }

    i32     HalfMeshFaceVertexCount( const HalfMesh & mesh, i32 face );
    void    HalfMeshFaceHalfEdges( const HalfMesh & mesh, i32 face, List<i32> & outHalfEdges );
    void    HalfMeshFaceVertices( const HalfMesh & mesh, i32 face, List<i32> & outVertices );
    // Every half-edge whose origin is this vertex, boundary ones included.
    void    HalfMeshVertexOutgoing( const HalfMesh & mesh, i32 vertex, List<i32> & outHalfEdges );
    i32     HalfMeshVertexValence( const HalfMesh & mesh, i32 vertex );

    Vec3    HalfMeshFaceCentroid( const HalfMesh & mesh, i32 face );
    // Newell's method, so a face that is not quite planar still gets a sane
    // normal instead of one taken from whichever three corners came first.
    Vec3    HalfMeshFaceNormal( const HalfMesh & mesh, i32 face );
    void    HalfMeshComputeFaceNormals( HalfMesh & mesh );
    Vec3    HalfMeshEdgeMidpoint( const HalfMesh & mesh, i32 edge );

    // --- editing ------------------------------------------------------------

    // These rebuild the mesh through the polygon soup rather than splicing the
    // connectivity in place. That costs O(mesh) per edit instead of O(1), which
    // is the wrong complexity for a sculpting brush and entirely fine for
    // clicking walls into a level. Both keep VERTEX and FACE indices stable, so
    // a selection survives an edit; EDGE and HALF-EDGE indices do not.

    // Pushes a face along its own normal and walls in the gap. Returns the face
    // index of the cap, which is the same index the original face had.
    i32 HalfMeshExtrudeFace( HalfMesh & mesh, i32 face, f32 distance );

    // Puts a vertex at t along the edge, 0 at its origin and 1 at its
    // destination. Both adjacent faces gain a side; neither is triangulated.
    // Returns the new vertex index, or kHMNone if the edge is invalid.
    i32 HalfMeshSplitEdge( HalfMesh & mesh, i32 edge, f32 t );

    // --- output -------------------------------------------------------------

    // Fan-triangulates every face, one set of vertices per face so the shading
    // stays flat. UVs are a planar projection along the face's dominant axis at
    // one unit per unit of world space, which lines a tiling texture up with the
    // grid.
    void HalfMeshTriangulate( const HalfMesh & mesh, List<HMTriVertex> & outVertices,
                              List<u32> & outIndices );

    // --- debugging ----------------------------------------------------------

    // Walks every invariant listed on HalfMesh. Cheap enough to call after each
    // edit while the editing code is still young; outError names the first thing
    // that was wrong and may be null.
    bool HalfMeshValidate( const HalfMesh & mesh, const char ** outError );

} // namespace sol
