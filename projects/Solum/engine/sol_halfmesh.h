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

    struct HalfMesh {
        List<HMVertex>  vertices;
        List<HMEdge>    edges;
        List<HMFace>    faces;
        List<HalfEdge>  halfEdges;
    };

    struct HMTriVertex {
        Vec3    position;
        Vec3    normal;
        Vec2    uv;
    };

    // --- lifetime -----------------------------------------------------------
    void        HalfMeshFree( HalfMesh & mesh );
    void        HalfMeshClear( HalfMesh & mesh );
    HalfMesh    HalfMeshCopy( const HalfMesh & mesh );

    // --- construction -------------------------------------------------------
    bool        HalfMeshFromPolygons( HalfMesh & mesh, const Vec3 * positions, i32 positionCount, const i32 * faceIndices, const i32 * faceSizes, i32 faceCount );
    void        HalfMeshToPolygons( const HalfMesh & mesh, List<Vec3> & outPositions, List<i32> & outFaceIndices, List<i32> & outFaceSizes );

    void        HalfMeshCreateTriangle( HalfMesh & mesh );
    void        HalfMeshCreateQuad( HalfMesh & mesh, f32 size );
    void        HalfMeshCreateCube( HalfMesh & mesh, i32 size );
    void        HalfMeshCreatePlane( HalfMesh & mesh, i32 cols, i32 rows, f32 cellSize );

    // --- traversal ----------------------------------------------------------
    inline i32  HalfEdgeDest( const HalfMesh & mesh, i32 halfEdge ) { return mesh.halfEdges[mesh.halfEdges[halfEdge].twin].vert; }
    inline bool HalfEdgeIsBoundary( const HalfMesh & mesh, i32 halfEdge ) { return mesh.halfEdges[halfEdge].face == kHMNone; }
    inline bool HalfMeshEdgeIsBoundary( const HalfMesh & mesh, i32 edge ) { const i32 h = mesh.edges[edge].halfEdge; return HalfEdgeIsBoundary( mesh, h ) || HalfEdgeIsBoundary( mesh, mesh.halfEdges[h].twin ); }
    inline i32  HalfEdgeRotateAroundVertex( const HalfMesh & mesh, i32 halfEdge ) { return mesh.halfEdges[mesh.halfEdges[halfEdge].twin].next; }
    i32         HalfEdgePrev( const HalfMesh & mesh, i32 halfEdge );

    i32         HalfMeshFaceVertexCount( const HalfMesh & mesh, i32 face );
    void        HalfMeshFaceHalfEdges( const HalfMesh & mesh, i32 face, List<i32> & outHalfEdges );
    void        HalfMeshFaceVertices( const HalfMesh & mesh, i32 face, List<i32> & outVertices );
    void        HalfMeshVertexOutgoing( const HalfMesh & mesh, i32 vertex, List<i32> & outHalfEdges );
    i32         HalfMeshVertexValence( const HalfMesh & mesh, i32 vertex );

    Vec3        HalfMeshFaceCentroid( const HalfMesh & mesh, i32 face );
    Vec3        HalfMeshFaceNormal( const HalfMesh & mesh, i32 face );
    void        HalfMeshComputeFaceNormals( HalfMesh & mesh );
    Vec3        HalfMeshEdgeMidpoint( const HalfMesh & mesh, i32 edge );

    // --- editing ------------------------------------------------------------
    i32 HalfMeshExtrudeFace( HalfMesh & mesh, i32 face, f32 distance );
    i32 HalfMeshSplitEdge( HalfMesh & mesh, i32 edge, f32 t );

    // --- output -------------------------------------------------------------
    void HalfMeshTriangulate( const HalfMesh & mesh, List<HMTriVertex> & outVertices, List<u32> & outIndices );

    // --- debugging ----------------------------------------------------------
    bool HalfMeshValidate( const HalfMesh & mesh, const char ** outError );

} // namespace sol
