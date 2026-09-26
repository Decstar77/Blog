#pragma once
#include "sol_defines.h"
#include "sol_list.h"
#include "sol_math.h"
#include "sol_string.h"

// Convex brushes, the way Quake-family editors build levels. A brush is the
// intersection of the half-spaces behind its planes. The planes are the
// authored data; every corner and polygon is derived from them by
// BrushRebuild, so two faces can never disagree about where their shared edge
// is, and nothing an edit does to the planes can leave a brush non-convex.

namespace sol {

    // World units. Two points closer than this are one point, and a point this
    // close to a plane is on it. The finest grid step sits far above it, so
    // snapped geometry never lands in the grey zone.
    constexpr f32 kBrushEpsilon = 1e-3f;

    struct Plane {
        Vec3    normal;     // unit, pointing out of the solid
        f32     distance;   // Vec3Dot( normal, p ) for every p on the plane
    };

    Plane   PlaneMake( Vec3 normal, Vec3 pointOnPlane );
    // Normal follows ( b - a ) x ( c - a ). False when the points are collinear.
    bool    PlaneFromPoints( Vec3 a, Vec3 b, Vec3 c, Plane * outPlane );
    Plane   PlaneFlip( Plane plane );
    // Signed: positive in front, which for a brush face means outside.
    f32     PlaneSide( Plane plane, Vec3 point );
    bool    PlaneEquals( Plane a, Plane b );

    // How a material is laid onto a face. Projected from world space along the
    // axis the face most faces (Quake's paraxial projection), so the faces of
    // neighbouring brushes line up without anyone aligning them.
    struct FaceTexture {
        SmallString     material;   // asset name relative to the asset directory, no extension; empty is untextured
        f32             offsetU;    // in texture repeats
        f32             offsetV;
        f32             scaleU;     // world units per texture repeat
        f32             scaleV;
        f32             rotation;   // degrees, turning the texture about the projection axis
    };

    // Zero scale would divide by zero, so start here rather than at {}.
    FaceTexture FaceTextureDefault();
    // World position to texture coordinates for a face pointing along normal.
    Vec2        FaceTextureUv( const FaceTexture & texture, Vec3 normal, Vec3 position );
    // The two world directions u and v advance along, already rotated and
    // scaled: u = dot( p, outU ) + offsetU.
    void        FaceTextureAxes( const FaceTexture & texture, Vec3 normal, Vec3 * outU, Vec3 * outV );
    // Texture lock for a move: shifts the offsets so the texture travels with
    // the face instead of the face sliding under a texture fixed in the world.
    void        FaceTextureLockTranslate( FaceTexture & texture, Vec3 normal, Vec3 delta );

    enum BrushFaceFlags : u32 {
        BrushFaceFlag_Selected  = 1u << 0,
    };

    struct BrushFace {
        Plane           plane;
        FaceTexture     texture;
        // Editor state. Carried on the face rather than beside it so an undo
        // snapshot puts the selection back along with the geometry. Neither
        // the runtime nor the file format reads it.
        u32             flags;
        // Derived by BrushRebuild: this face's corners in Brush::points,
        // counter-clockwise seen from outside.
        i32             firstPoint;
        i32             pointCount;
    };

    enum BrushFlags : u32 {
        BrushFlag_Selected      = 1u << 0,
        BrushFlag_Hidden        = 1u << 1,
    };

    struct Brush {
        List<BrushFace>     faces;
        // Derived: every face's polygon back to back. A corner shared by three
        // faces is stored three times, welded to identical bits so the copies
        // compare exactly.
        List<Vec3>          points;
        u32                 flags;      // BrushFlags, editor state like BrushFace::flags
    };

    void    BrushFree( Brush & brush );
    Brush   BrushCopy( const Brush & brush );

    // Derives the polygons from the planes. Duplicate planes, and planes the
    // others already cut away entirely, are dropped, so face indices can shift.
    // False when what is left encloses no volume.
    bool    BrushRebuild( Brush & brush );

    bool    BrushCreateBox( Brush & brush, Vec3 min, Vec3 max, const FaceTexture & texture );
    // Convex hull of points, one face per hull plane. Each face takes its
    // texture from whichever template face points the most the same way, which
    // is what keeps materials in place through a vertex edit. False when the
    // points are flat.
    bool    BrushCreateHull( Brush & brush, const Vec3 * points, i32 pointCount, const BrushFace * templates, i32 templateCount, const FaceTexture & fallback );
    // A prism swept out of one face of source along its normal. Its sides take
    // the textures of the source faces they continue.
    bool    BrushCreateExtrusion( Brush & brush, const Brush & source, i32 face, f32 distance );

    void    BrushBounds( const Brush & brush, Vec3 * outMin, Vec3 * outMax );
    // Every distinct corner once.
    void    BrushVertices( const Brush & brush, List<Vec3> & outVertices );
    // Every edge once, as consecutive pairs of end points.
    void    BrushEdges( const Brush & brush, List<Vec3> & outSegments );
    Vec3    BrushFaceCenter( const Brush & brush, i32 face );
    // The texture of the face whose normal is nearest direction, for surfaces
    // an edit creates and something has to cover.
    FaceTexture BrushNearestTexture( const Brush & brush, Vec3 direction );

    // Texture lock keeps each face's texture riding along with it.
    void    BrushTranslate( Brush & brush, Vec3 delta, bool textureLock );
    // transform must be rigid, or a mirror: rotations and flips only. Normals
    // that land within a hair of an axis are put exactly on it, so a box turned
    // by 90 degrees is still an exact box.
    bool    BrushTransform( Brush & brush, const Mat4 & transform );
    // Slides one face along its normal. Fails, leaving the brush untouched,
    // when that would turn the brush inside out or make any face vanish.
    bool    BrushMoveFace( Brush & brush, i32 face, f32 distance );

    // The part of brush behind plane, capped by a new face wearing capTexture.
    // A plane that misses the brush leaves it whole on one side: behind, and
    // this returns a copy; in front, and it returns false.
    bool    BrushClipBehind( const Brush & brush, Plane plane, const FaceTexture & capTexture, Brush * outBrush );
    // True only for overlapping volume. Brushes that merely share a face do
    // not intersect.
    bool    BrushIntersects( const Brush & a, const Brush & b );
    // Appends the convex pieces of a that lie outside b. The walls the cut
    // exposes wear the textures of the faces of b that made them.
    void    BrushSubtract( const Brush & a, const Brush & b, List<Brush> & outPieces );

    // Nearest face the ray enters. A ray starting inside hits the face it
    // leaves through, so a brush the camera is standing in is still pickable.
    bool    BrushRaycast( const Brush & brush, Vec3 origin, Vec3 direction, f32 * outDistance, i32 * outFace );

} // namespace sol
