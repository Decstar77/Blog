#include "sol_brush.h"

#include <cmath>

namespace sol {

    // Polygons are cut out of a huge quad on each plane, so the first few
    // clips subtract numbers this size. Doubles keep the corners that survive
    // accurate to far below kBrushEpsilon; floats would not.
    struct DVec3 {
        f64 x;
        f64 y;
        f64 z;
    };

    static DVec3 DVec3From( Vec3 v ) { return DVec3{ v.x, v.y, v.z }; }
    static DVec3 DAdd( DVec3 a, DVec3 b ) { return DVec3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
    static DVec3 DSub( DVec3 a, DVec3 b ) { return DVec3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
    static DVec3 DScale( DVec3 v, f64 s ) { return DVec3{ v.x * s, v.y * s, v.z * s }; }
    static f64 DDot( DVec3 a, DVec3 b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    static DVec3 DCross( DVec3 a, DVec3 b ) { return DVec3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
    static f64 DLength( DVec3 v ) { return sqrt( DDot( v, v ) ); }

    // How big the quad each face polygon is cut from is, either side of the
    // plane's origin. Anything a level puts further out than this is clipped.
    constexpr f64 kBrushWorldHalfExtent = 65536.0;

    // Corners that come out of the solve a rounding error away from a 1/1024
    // lattice point are put exactly on it. That is every corner of a brush on
    // the grid, and it is what lets the same corner computed from two different
    // triples of planes come out as the same bits.
    static f32 BrushRoundCoordinate( f64 value ) {
        const f64 snapped = floor( value * 1024.0 + 0.5 ) / 1024.0;
        return (f32)( fabs( value - snapped ) < 2e-5 ? snapped : value );
    }

    static Vec3 BrushRoundPoint( DVec3 p ) {
        return Vec3{ BrushRoundCoordinate( p.x ), BrushRoundCoordinate( p.y ), BrushRoundCoordinate( p.z ) };
    }

    static bool Vec3Near( Vec3 a, Vec3 b, f32 epsilon ) {
        return fabsf( a.x - b.x ) <= epsilon && fabsf( a.y - b.y ) <= epsilon && fabsf( a.z - b.z ) <= epsilon;
    }

    Plane PlaneMake( Vec3 normal, Vec3 pointOnPlane ) {
        Plane plane = {};
        plane.normal = Vec3Normalize( normal );
        plane.distance = Vec3Dot( plane.normal, pointOnPlane );
        return plane;
    }

    bool PlaneFromPoints( Vec3 a, Vec3 b, Vec3 c, Plane * outPlane ) {
        const DVec3 da = DVec3From( a );
        const DVec3 cross = DCross( DSub( DVec3From( b ), da ), DSub( DVec3From( c ), da ) );
        const f64 length = DLength( cross );
        if( length < 1e-9 ) {
            return false;
        }
        const DVec3 normal = DScale( cross, 1.0 / length );
        outPlane->normal = Vec3{ (f32)normal.x, (f32)normal.y, (f32)normal.z };
        outPlane->distance = (f32)DDot( normal, da );
        return true;
    }

    Plane PlaneFlip( Plane plane ) {
        return Plane{ plane.normal * -1.0f, -plane.distance };
    }

    f32 PlaneSide( Plane plane, Vec3 point ) {
        return Vec3Dot( plane.normal, point ) - plane.distance;
    }

    bool PlaneEquals( Plane a, Plane b ) {
        return Vec3Dot( a.normal, b.normal ) > 1.0f - 1e-5f && fabsf( a.distance - b.distance ) < kBrushEpsilon;
    }

    // --- texture projection -------------------------------------------------

    FaceTexture FaceTextureDefault() {
        FaceTexture texture = {};
        texture.scaleU = 1.0f;
        texture.scaleV = 1.0f;
        return texture;
    }

    // The unrotated axes for a face. Floors read the way the top view shows
    // them, and walls read upright from outside: v runs down because image
    // rows do.
    static void FaceTextureBaseAxes( Vec3 normal, Vec3 * outU, Vec3 * outV ) {
        const f32 ax = fabsf( normal.x );
        const f32 ay = fabsf( normal.y );
        const f32 az = fabsf( normal.z );
        if( ay >= ax && ay >= az ) {
            *outU = Vec3{ 1.0f, 0.0f, 0.0f };
            *outV = Vec3{ 0.0f, 0.0f, 1.0f };
        } else if( ax >= az ) {
            *outU = Vec3{ 0.0f, 0.0f, normal.x > 0.0f ? -1.0f : 1.0f };
            *outV = Vec3{ 0.0f, -1.0f, 0.0f };
        } else {
            *outU = Vec3{ normal.z > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f };
            *outV = Vec3{ 0.0f, -1.0f, 0.0f };
        }
    }

    void FaceTextureAxes( const FaceTexture & texture, Vec3 normal, Vec3 * outU, Vec3 * outV ) {
        Vec3 u = {};
        Vec3 v = {};
        FaceTextureBaseAxes( normal, &u, &v );

        const f32 radians = texture.rotation * kDeg2Rad;
        const f32 c = cosf( radians );
        const f32 s = sinf( radians );
        const Vec3 rotatedU = u * c + v * s;
        const Vec3 rotatedV = v * c - u * s;

        // A zero scale would put the whole texture into a point; read it as
        // one repeat per unit instead of dividing by it.
        const f32 scaleU = texture.scaleU != 0.0f ? texture.scaleU : 1.0f;
        const f32 scaleV = texture.scaleV != 0.0f ? texture.scaleV : 1.0f;
        *outU = rotatedU * ( 1.0f / scaleU );
        *outV = rotatedV * ( 1.0f / scaleV );
    }

    Vec2 FaceTextureUv( const FaceTexture & texture, Vec3 normal, Vec3 position ) {
        Vec3 u = {};
        Vec3 v = {};
        FaceTextureAxes( texture, normal, &u, &v );
        return Vec2{ Vec3Dot( position, u ) + texture.offsetU, Vec3Dot( position, v ) + texture.offsetV };
    }

    void FaceTextureLockTranslate( FaceTexture & texture, Vec3 normal, Vec3 delta ) {
        Vec3 u = {};
        Vec3 v = {};
        FaceTextureAxes( texture, normal, &u, &v );
        texture.offsetU -= Vec3Dot( delta, u );
        texture.offsetV -= Vec3Dot( delta, v );
        // Whole repeats change nothing on screen, and dropping them keeps the
        // numbers the inspector shows small after a long run of moves.
        texture.offsetU -= floorf( texture.offsetU );
        texture.offsetV -= floorf( texture.offsetV );
    }

    // --- lifetime -----------------------------------------------------------

    void BrushFree( Brush & brush ) {
        ListFree( brush.faces );
        ListFree( brush.points );
        brush.flags = 0;
    }

    Brush BrushCopy( const Brush & brush ) {
        Brush copy = {};
        copy.faces = ListCopy( brush.faces );
        copy.points = ListCopy( brush.points );
        copy.flags = brush.flags;
        return copy;
    }

    // --- derivation ---------------------------------------------------------

    // Keeps the part of polygon behind plane, with anything within
    // kBrushEpsilon counted as on it. Points on the plane are kept but never
    // split against, so a polygon lying in the plane passes through whole.
    static void ClipPolygon( const List<DVec3> & polygon, DVec3 normal, f64 distance, List<DVec3> & outPolygon ) {
        ListClear( outPolygon );
        const f64 epsilon = (f64)kBrushEpsilon;
        for( i32 i = 0; i < polygon.count; i++ ) {
            const DVec3 a = polygon[i];
            const DVec3 b = polygon[( i + 1 ) % polygon.count];
            const f64 sideA = DDot( normal, a ) - distance;
            const f64 sideB = DDot( normal, b ) - distance;

            if( sideA <= epsilon ) {
                ListAdd( outPolygon, a );
            }
            // Only a clean crossing makes a new corner. An end already on the
            // plane is kept by its own turn through the loop.
            if( ( sideA < -epsilon && sideB > epsilon ) || ( sideA > epsilon && sideB < -epsilon ) ) {
                const f64 t = sideA / ( sideA - sideB );
                ListAdd( outPolygon, DAdd( a, DScale( DSub( b, a ), t ) ) );
            }
        }
    }

    bool BrushRebuild( Brush & brush ) {
        // A plane listed twice would produce the same polygon twice.
        for( i32 i = 1; i < brush.faces.count; i++ ) {
            for( i32 j = 0; j < i; j++ ) {
                if( PlaneEquals( brush.faces[i].plane, brush.faces[j].plane ) ) {
                    ListRemoveIndex( brush.faces, i );
                    i--;
                    break;
                }
            }
        }

        ListClear( brush.points );

        List<DVec3> polygon = {};
        List<DVec3> clipped = {};

        for( i32 f = 0; f < brush.faces.count; f++ ) {
            BrushFace & face = brush.faces[f];
            face.firstPoint = brush.points.count;
            face.pointCount = 0;

            const DVec3 normal = DScale( DVec3From( face.plane.normal ), 1.0 / Max( Vec3Length( face.plane.normal ), 1e-12f ) );
            const f64 distance = (f64)face.plane.distance;

            // u x v = normal, so walking the quad u-v, then round, runs
            // counter-clockwise seen from outside, and clipping keeps that order.
            const DVec3 helper = fabs( normal.x ) < 0.9 ? DVec3{ 1.0, 0.0, 0.0 } : DVec3{ 0.0, 1.0, 0.0 };
            DVec3 u = DCross( helper, normal );
            u = DScale( u, 1.0 / DLength( u ) );
            const DVec3 v = DCross( normal, u );
            const DVec3 center = DScale( normal, distance );
            const DVec3 us = DScale( u, kBrushWorldHalfExtent );
            const DVec3 vs = DScale( v, kBrushWorldHalfExtent );

            ListClear( polygon );
            ListAdd( polygon, DSub( DSub( center, us ), vs ) );
            ListAdd( polygon, DSub( DAdd( center, us ), vs ) );
            ListAdd( polygon, DAdd( DAdd( center, us ), vs ) );
            ListAdd( polygon, DAdd( DSub( center, us ), vs ) );

            for( i32 other = 0; other < brush.faces.count && polygon.count >= 3; other++ ) {
                if( other == f ) {
                    continue;
                }
                const Plane & plane = brush.faces[other].plane;
                ClipPolygon( polygon, DVec3From( plane.normal ), (f64)plane.distance, clipped );

                List<DVec3> swap = polygon;
                polygon = clipped;
                clipped = swap;
            }

            // Grazing clips leave corners on top of each other.
            for( i32 i = 0; i < polygon.count && polygon.count >= 3; i++ ) {
                const DVec3 next = polygon[( i + 1 ) % polygon.count];
                if( DLength( DSub( polygon[i], next ) ) < 1e-5 ) {
                    ListRemoveIndex( polygon, i );
                    i--;
                }
            }

            if( polygon.count < 3 ) {
                continue;
            }

            for( i32 i = 0; i < polygon.count; i++ ) {
                ListAdd( brush.points, BrushRoundPoint( polygon[i] ) );
            }
            face.pointCount = polygon.count;
        }

        ListFree( polygon );
        ListFree( clipped );

        // Welded to the first copy, so a corner shared by several faces is the
        // same bits in all of them and the tools can match corners exactly.
        for( i32 i = 1; i < brush.points.count; i++ ) {
            for( i32 j = 0; j < i; j++ ) {
                if( Vec3Near( brush.points[i], brush.points[j], 1e-4f ) ) {
                    brush.points[i] = brush.points[j];
                    break;
                }
            }
        }

        // A face whose polygon was cut away completely is a plane the others
        // already make redundant.
        ListRemoveIf( brush.faces, []( const BrushFace & face ) { return face.pointCount < 3; } );

        if( brush.faces.count < 4 ) {
            return false;
        }

        // A solid has thickness along every world axis. A brush squeezed flat
        // can keep four slivers of faces, so this is checked as well as count.
        Vec3 min = {};
        Vec3 max = {};
        BrushBounds( brush, &min, &max );
        return max.x - min.x > kBrushEpsilon && max.y - min.y > kBrushEpsilon && max.z - min.z > kBrushEpsilon;
    }

    // --- construction -------------------------------------------------------

    static void BrushAddFace( Brush & brush, Plane plane, const FaceTexture & texture ) {
        BrushFace face = {};
        face.plane = plane;
        face.texture = texture;
        ListAdd( brush.faces, face );
    }

    bool BrushCreateBox( Brush & brush, Vec3 min, Vec3 max, const FaceTexture & texture ) {
        BrushFree( brush );
        BrushAddFace( brush, Plane{ Vec3{  1.0f, 0.0f, 0.0f },  max.x }, texture );
        BrushAddFace( brush, Plane{ Vec3{ -1.0f, 0.0f, 0.0f }, -min.x }, texture );
        BrushAddFace( brush, Plane{ Vec3{ 0.0f,  1.0f, 0.0f },  max.y }, texture );
        BrushAddFace( brush, Plane{ Vec3{ 0.0f, -1.0f, 0.0f }, -min.y }, texture );
        BrushAddFace( brush, Plane{ Vec3{ 0.0f, 0.0f,  1.0f },  max.z }, texture );
        BrushAddFace( brush, Plane{ Vec3{ 0.0f, 0.0f, -1.0f }, -min.z }, texture );
        return BrushRebuild( brush );
    }

    static const FaceTexture * NearestTemplate( const BrushFace * templates, i32 templateCount, Vec3 normal ) {
        const FaceTexture * best = nullptr;
        f32 bestDot = -2.0f;
        for( i32 i = 0; i < templateCount; i++ ) {
            const f32 d = Vec3Dot( templates[i].plane.normal, normal );
            if( d > bestDot ) {
                bestDot = d;
                best = &templates[i].texture;
            }
        }
        return best;
    }

    bool BrushCreateHull( Brush & brush, const Vec3 * points, i32 pointCount, const BrushFace * templates, i32 templateCount, const FaceTexture & fallback ) {
        BrushFree( brush );

        List<DVec3> unique = {};
        for( i32 i = 0; i < pointCount; i++ ) {
            bool seen = false;
            for( i32 j = 0; j < i && !seen; j++ ) {
                seen = Vec3Near( points[i], points[j], kBrushEpsilon );
            }
            if( !seen ) {
                ListAdd( unique, DVec3From( points[i] ) );
            }
        }

        List<Plane> planes = {};
        const f64 epsilon = (f64)kBrushEpsilon;

        // Every triple of points spans a candidate plane, and it is a hull face
        // exactly when every other point is on one side of it. Cubic in the
        // point count, which a brush keeps small, and immune to the ordering
        // and coplanarity traps an incremental hull has to be careful of.
        for( i32 i = 0; i < unique.count; i++ ) {
            for( i32 j = i + 1; j < unique.count; j++ ) {
                const DVec3 ab = DSub( unique[j], unique[i] );
                for( i32 k = j + 1; k < unique.count; k++ ) {
                    const DVec3 cross = DCross( ab, DSub( unique[k], unique[i] ) );
                    const f64 length = DLength( cross );
                    if( length < 1e-9 ) {
                        continue;
                    }
                    const DVec3 normal = DScale( cross, 1.0 / length );
                    const f64 distance = DDot( normal, unique[i] );

                    bool above = false;
                    bool below = false;
                    for( i32 m = 0; m < unique.count && !( above && below ); m++ ) {
                        const f64 side = DDot( normal, unique[m] ) - distance;
                        if( side > epsilon ) {
                            above = true;
                        } else if( side < -epsilon ) {
                            below = true;
                        }
                    }
                    // Points on both sides: an inner plane. On neither: every
                    // point is in it, which is a flat set with no hull at all.
                    if( above == below ) {
                        continue;
                    }

                    Plane plane = {};
                    plane.normal = Vec3{ (f32)normal.x, (f32)normal.y, (f32)normal.z };
                    plane.distance = (f32)distance;
                    if( above ) {
                        plane = PlaneFlip( plane );
                    }

                    bool duplicate = false;
                    for( i32 p = 0; p < planes.count && !duplicate; p++ ) {
                        duplicate = PlaneEquals( planes[p], plane );
                    }
                    if( !duplicate ) {
                        ListAdd( planes, plane );
                    }
                }
            }
        }

        for( i32 p = 0; p < planes.count; p++ ) {
            const FaceTexture * texture = NearestTemplate( templates, templateCount, planes[p].normal );
            BrushAddFace( brush, planes[p], texture != nullptr ? *texture : fallback );
        }

        ListFree( unique );
        ListFree( planes );
        return BrushRebuild( brush );
    }

    // Whether face holds both a and b among its corners, which for a convex
    // brush means the edge between them is one of its sides.
    static bool FaceHasEdge( const Brush & brush, i32 face, Vec3 a, Vec3 b ) {
        const BrushFace & f = brush.faces[face];
        bool hasA = false;
        bool hasB = false;
        for( i32 i = 0; i < f.pointCount; i++ ) {
            const Vec3 p = brush.points[f.firstPoint + i];
            hasA = hasA || Vec3Near( p, a, kBrushEpsilon );
            hasB = hasB || Vec3Near( p, b, kBrushEpsilon );
        }
        return hasA && hasB;
    }

    bool BrushCreateExtrusion( Brush & brush, const Brush & source, i32 face, f32 distance ) {
        BrushFree( brush );
        if( face < 0 || face >= source.faces.count || distance <= kBrushEpsilon ) {
            return false;
        }

        const BrushFace & base = source.faces[face];
        const Vec3 normal = base.plane.normal;

        BrushAddFace( brush, Plane{ normal, base.plane.distance + distance }, base.texture );
        BrushAddFace( brush, PlaneFlip( base.plane ), base.texture );

        for( i32 i = 0; i < base.pointCount; i++ ) {
            const Vec3 a = source.points[base.firstPoint + i];
            const Vec3 b = source.points[base.firstPoint + ( i + 1 ) % base.pointCount];

            // The polygon runs counter-clockwise about the normal, so the
            // outside of each edge is to its right: edge x normal.
            const Vec3 sideNormal = Vec3Cross( b - a, normal );
            if( Vec3Length( sideNormal ) < 1e-6f ) {
                continue;
            }

            FaceTexture texture = base.texture;
            for( i32 other = 0; other < source.faces.count; other++ ) {
                if( other != face && FaceHasEdge( source, other, a, b ) ) {
                    texture = source.faces[other].texture;
                    break;
                }
            }
            BrushAddFace( brush, PlaneMake( sideNormal, a ), texture );
        }

        return BrushRebuild( brush );
    }

    // --- queries ------------------------------------------------------------

    void BrushBounds( const Brush & brush, Vec3 * outMin, Vec3 * outMax ) {
        if( brush.points.count == 0 ) {
            *outMin = Vec3{};
            *outMax = Vec3{};
            return;
        }
        Vec3 min = brush.points[0];
        Vec3 max = brush.points[0];
        for( i32 i = 1; i < brush.points.count; i++ ) {
            const Vec3 p = brush.points[i];
            min = Vec3{ Min( min.x, p.x ), Min( min.y, p.y ), Min( min.z, p.z ) };
            max = Vec3{ Max( max.x, p.x ), Max( max.y, p.y ), Max( max.z, p.z ) };
        }
        *outMin = min;
        *outMax = max;
    }

    void BrushVertices( const Brush & brush, List<Vec3> & outVertices ) {
        ListClear( outVertices );
        for( i32 i = 0; i < brush.points.count; i++ ) {
            bool seen = false;
            for( i32 j = 0; j < outVertices.count && !seen; j++ ) {
                seen = Vec3Near( outVertices[j], brush.points[i], kBrushEpsilon );
            }
            if( !seen ) {
                ListAdd( outVertices, brush.points[i] );
            }
        }
    }

    void BrushEdges( const Brush & brush, List<Vec3> & outSegments ) {
        ListClear( outSegments );
        for( i32 f = 0; f < brush.faces.count; f++ ) {
            const BrushFace & face = brush.faces[f];
            for( i32 i = 0; i < face.pointCount; i++ ) {
                const Vec3 a = brush.points[face.firstPoint + i];
                const Vec3 b = brush.points[face.firstPoint + ( i + 1 ) % face.pointCount];

                // Each edge borders two faces and turns up once from each side.
                bool seen = false;
                for( i32 s = 0; s + 1 < outSegments.count && !seen; s += 2 ) {
                    seen = ( Vec3Near( outSegments[s], b, kBrushEpsilon ) && Vec3Near( outSegments[s + 1], a, kBrushEpsilon ) ) ||
                           ( Vec3Near( outSegments[s], a, kBrushEpsilon ) && Vec3Near( outSegments[s + 1], b, kBrushEpsilon ) );
                }
                if( !seen ) {
                    ListAdd( outSegments, a );
                    ListAdd( outSegments, b );
                }
            }
        }
    }

    Vec3 BrushFaceCenter( const Brush & brush, i32 face ) {
        const BrushFace & f = brush.faces[face];
        Vec3 sum = {};
        for( i32 i = 0; i < f.pointCount; i++ ) {
            sum = sum + brush.points[f.firstPoint + i];
        }
        return f.pointCount > 0 ? sum * ( 1.0f / (f32)f.pointCount ) : sum;
    }

    FaceTexture BrushNearestTexture( const Brush & brush, Vec3 direction ) {
        const FaceTexture * texture = NearestTemplate( brush.faces.data, brush.faces.count, direction );
        return texture != nullptr ? *texture : FaceTextureDefault();
    }

    // --- edits --------------------------------------------------------------

    void BrushTranslate( Brush & brush, Vec3 delta, bool textureLock ) {
        for( i32 f = 0; f < brush.faces.count; f++ ) {
            BrushFace & face = brush.faces[f];
            face.plane.distance += Vec3Dot( face.plane.normal, delta );
            if( textureLock ) {
                FaceTextureLockTranslate( face.texture, face.plane.normal, delta );
            }
        }
        // A pure translation cannot change which faces exist, so the corners
        // move with it rather than being solved again.
        for( i32 i = 0; i < brush.points.count; i++ ) {
            brush.points[i] = brush.points[i] + delta;
        }
    }

    static f32 CleanAxisComponent( f32 value ) {
        if( fabsf( value ) < 1e-6f ) {
            return 0.0f;
        }
        if( fabsf( value - 1.0f ) < 1e-6f ) {
            return 1.0f;
        }
        if( fabsf( value + 1.0f ) < 1e-6f ) {
            return -1.0f;
        }
        return value;
    }

    bool BrushTransform( Brush & brush, const Mat4 & transform ) {
        Brush result = BrushCopy( brush );
        for( i32 f = 0; f < result.faces.count; f++ ) {
            BrushFace & face = result.faces[f];
            const Vec3 onPlane = face.plane.normal * face.plane.distance;
            const Vec3 moved = Mat4MulPoint( transform, onPlane );

            Vec3 normal = Mat4MulDir( transform, face.plane.normal );
            normal = Vec3Normalize( Vec3{ CleanAxisComponent( normal.x ), CleanAxisComponent( normal.y ), CleanAxisComponent( normal.z ) } );

            face.plane.normal = normal;
            face.plane.distance = BrushRoundCoordinate( (f64)Vec3Dot( normal, moved ) );
        }

        if( !BrushRebuild( result ) ) {
            BrushFree( result );
            return false;
        }
        BrushFree( brush );
        brush = result;
        return true;
    }

    bool BrushMoveFace( Brush & brush, i32 face, f32 distance ) {
        if( face < 0 || face >= brush.faces.count ) {
            return false;
        }

        Brush result = BrushCopy( brush );
        result.faces[face].plane.distance += distance;
        // The face count staying put is what says no face was squeezed out.
        // Without it a drag could quietly delete the face next to the one moved.
        if( !BrushRebuild( result ) || result.faces.count != brush.faces.count ) {
            BrushFree( result );
            return false;
        }
        BrushFree( brush );
        brush = result;
        return true;
    }

    bool BrushClipBehind( const Brush & brush, Plane plane, const FaceTexture & capTexture, Brush * outBrush ) {
        Brush result = BrushCopy( brush );
        BrushAddFace( result, plane, capTexture );
        if( !BrushRebuild( result ) ) {
            BrushFree( result );
            return false;
        }
        *outBrush = result;
        return true;
    }

    static bool BoundsOverlap( const Brush & a, const Brush & b ) {
        Vec3 aMin = {};
        Vec3 aMax = {};
        Vec3 bMin = {};
        Vec3 bMax = {};
        BrushBounds( a, &aMin, &aMax );
        BrushBounds( b, &bMin, &bMax );
        return aMin.x < bMax.x - kBrushEpsilon && bMin.x < aMax.x - kBrushEpsilon &&
               aMin.y < bMax.y - kBrushEpsilon && bMin.y < aMax.y - kBrushEpsilon &&
               aMin.z < bMax.z - kBrushEpsilon && bMin.z < aMax.z - kBrushEpsilon;
    }

    bool BrushIntersects( const Brush & a, const Brush & b ) {
        if( !BoundsOverlap( a, b ) ) {
            return false;
        }

        Brush remaining = BrushCopy( a );
        for( i32 f = 0; f < b.faces.count; f++ ) {
            Brush clipped = {};
            const bool ok = BrushClipBehind( remaining, b.faces[f].plane, b.faces[f].texture, &clipped );
            BrushFree( remaining );
            if( !ok ) {
                return false;
            }
            remaining = clipped;
        }
        BrushFree( remaining );
        return true;
    }

    void BrushSubtract( const Brush & a, const Brush & b, List<Brush> & outPieces ) {
        if( !BrushIntersects( a, b ) ) {
            ListAdd( outPieces, BrushCopy( a ) );
            return;
        }

        // Peel a apart one face of b at a time: whatever lies in front of a
        // face of b is outside b, and is a finished convex piece. What lies
        // behind every face is inside b, and is thrown away.
        Brush remaining = BrushCopy( a );
        for( i32 f = 0; f < b.faces.count; f++ ) {
            const Plane plane = b.faces[f].plane;

            Brush outside = {};
            if( BrushClipBehind( remaining, PlaneFlip( plane ), b.faces[f].texture, &outside ) ) {
                ListAdd( outPieces, outside );
            }

            Brush inside = {};
            const bool any = BrushClipBehind( remaining, plane, b.faces[f].texture, &inside );
            BrushFree( remaining );
            if( !any ) {
                return;
            }
            remaining = inside;
        }
        BrushFree( remaining );
    }

    bool BrushRaycast( const Brush & brush, Vec3 origin, Vec3 direction, f32 * outDistance, i32 * outFace ) {
        f32 enter = -3.4e38f;
        f32 exit = 3.4e38f;
        i32 enterFace = -1;
        i32 exitFace = -1;

        for( i32 f = 0; f < brush.faces.count; f++ ) {
            const Plane & plane = brush.faces[f].plane;
            const f32 denom = Vec3Dot( plane.normal, direction );
            const f32 side = PlaneSide( plane, origin );

            if( fabsf( denom ) < 1e-9f ) {
                // Running parallel to a face from outside it never gets in.
                if( side > 0.0f ) {
                    return false;
                }
                continue;
            }

            const f32 t = -side / denom;
            if( denom < 0.0f ) {
                if( t > enter ) {
                    enter = t;
                    enterFace = f;
                }
            } else {
                if( t < exit ) {
                    exit = t;
                    exitFace = f;
                }
            }
        }

        if( enter > exit || exit < 0.0f ) {
            return false;
        }

        if( enter >= 0.0f && enterFace >= 0 ) {
            *outDistance = enter;
            *outFace = enterFace;
        } else {
            *outDistance = exit;
            *outFace = exitFace;
        }
        return *outFace >= 0;
    }

} // namespace sol
