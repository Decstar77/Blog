#include "sol_editor_doc.h"

namespace sol {

    static bool BrushVisible( const Brush & brush ) {
        return ( brush.flags & BrushFlag_Hidden ) == 0;
    }

    static void ClearFaceFlags( Brush & brush ) {
        for( i32 f = 0; f < brush.faces.count; f++ ) {
            brush.faces[f].flags &= ~BrushFaceFlag_Selected;
        }
    }

    // Swaps the brushes of the document for newBrushes, freeing the old ones.
    // Every multi-brush edit builds its result off to the side and lands it
    // here, so a failure halfway leaves the document as it was.
    static void ReplaceBrushes( EditorDoc & doc, List<Brush> & newBrushes ) {
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            BrushFree( doc.map.brushes[i] );
        }
        ListFree( doc.map.brushes );
        doc.map.brushes = newBrushes;
        newBrushes = {};
    }

    static void FreeBrushList( List<Brush> & brushes ) {
        for( i32 i = 0; i < brushes.count; i++ ) {
            BrushFree( brushes[i] );
        }
        ListFree( brushes );
    }

    static void FreeHistory( List<Map> & history ) {
        for( i32 i = 0; i < history.count; i++ ) {
            MapFree( history[i] );
        }
        ListFree( history );
    }

    EditorDoc DocCreate() {
        EditorDoc doc = {};
        doc.version = 1;
        return doc;
    }

    void DocFree( EditorDoc & doc ) {
        MapFree( doc.map );
        FreeHistory( doc.undo );
        FreeHistory( doc.redo );
        doc = {};
    }

    void DocReplaceMap( EditorDoc & doc, Map & map ) {
        const u32 version = doc.version;
        DocFree( doc );
        doc.map = map;
        map = {};
        doc.version = version + 1;
        doc.modified = false;
    }

    // --- history ------------------------------------------------------------

    void DocBeginEdit( EditorDoc & doc ) {
        if( doc.undo.count >= kDocMaxUndo ) {
            MapFree( doc.undo[0] );
            ListRemoveIndex( doc.undo, 0 );
        }
        ListAdd( doc.undo, MapCopy( doc.map ) );
        // A new edit forks history; what was undone can no longer be redone
        // onto a map that has gone somewhere else.
        FreeHistory( doc.redo );
        doc.modifiedBeforeEdit = doc.modified;
    }

    void DocEdited( EditorDoc & doc ) {
        doc.version++;
        doc.modified = true;
    }

    void DocAbandonEdit( EditorDoc & doc ) {
        if( doc.undo.count == 0 ) {
            return;
        }
        MapFree( doc.map );
        doc.map = doc.undo[doc.undo.count - 1];
        doc.undo.count--;
        doc.version++;
        doc.modified = doc.modifiedBeforeEdit;
    }

    void DocTouch( EditorDoc & doc ) {
        doc.version++;
    }

    void DocSelectionChanged( EditorDoc & doc ) {
        doc.version++;
    }

    bool DocUndo( EditorDoc & doc ) {
        if( doc.undo.count == 0 ) {
            return false;
        }
        ListAdd( doc.redo, doc.map );
        doc.map = doc.undo[doc.undo.count - 1];
        doc.undo.count--;
        doc.version++;
        doc.modified = true;
        return true;
    }

    bool DocRedo( EditorDoc & doc ) {
        if( doc.redo.count == 0 ) {
            return false;
        }
        ListAdd( doc.undo, doc.map );
        doc.map = doc.redo[doc.redo.count - 1];
        doc.redo.count--;
        doc.version++;
        doc.modified = true;
        return true;
    }

    // --- selection ----------------------------------------------------------

    bool DocIsSelected( const EditorDoc & doc, i32 brush ) {
        return brush >= 0 && brush < doc.map.brushes.count && ( doc.map.brushes[brush].flags & BrushFlag_Selected ) != 0;
    }

    i32 DocSelectedCount( const EditorDoc & doc ) {
        i32 count = 0;
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            count += DocIsSelected( doc, i ) ? 1 : 0;
        }
        return count;
    }

    void DocSelectedBrushes( const EditorDoc & doc, List<i32> & outIndices ) {
        ListClear( outIndices );
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            if( DocIsSelected( doc, i ) ) {
                ListAdd( outIndices, i );
            }
        }
    }

    void DocSetSelected( EditorDoc & doc, i32 brush, bool selected ) {
        if( brush < 0 || brush >= doc.map.brushes.count ) {
            return;
        }
        if( selected ) {
            for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
                ClearFaceFlags( doc.map.brushes[i] );
            }
            doc.map.brushes[brush].flags |= BrushFlag_Selected;
        } else {
            doc.map.brushes[brush].flags &= ~BrushFlag_Selected;
        }
        DocSelectionChanged( doc );
    }

    void DocSelectOnly( EditorDoc & doc, i32 brush ) {
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            doc.map.brushes[i].flags &= ~BrushFlag_Selected;
            ClearFaceFlags( doc.map.brushes[i] );
        }
        if( brush >= 0 && brush < doc.map.brushes.count ) {
            doc.map.brushes[brush].flags |= BrushFlag_Selected;
        }
        DocSelectionChanged( doc );
    }

    void DocSelectAll( EditorDoc & doc ) {
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            Brush & brush = doc.map.brushes[i];
            ClearFaceFlags( brush );
            if( BrushVisible( brush ) ) {
                brush.flags |= BrushFlag_Selected;
            }
        }
        DocSelectionChanged( doc );
    }

    void DocSelectNone( EditorDoc & doc ) {
        DocSelectOnly( doc, kNoBrush );
    }

    i32 DocSelectedFaceCount( const EditorDoc & doc ) {
        i32 count = 0;
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            const Brush & brush = doc.map.brushes[i];
            for( i32 f = 0; f < brush.faces.count; f++ ) {
                count += ( brush.faces[f].flags & BrushFaceFlag_Selected ) ? 1 : 0;
            }
        }
        return count;
    }

    void DocSetFaceSelected( EditorDoc & doc, i32 brush, i32 face, bool selected ) {
        if( brush < 0 || brush >= doc.map.brushes.count || face < 0 || face >= doc.map.brushes[brush].faces.count ) {
            return;
        }
        if( selected ) {
            for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
                doc.map.brushes[i].flags &= ~BrushFlag_Selected;
            }
            doc.map.brushes[brush].faces[face].flags |= BrushFaceFlag_Selected;
        } else {
            doc.map.brushes[brush].faces[face].flags &= ~BrushFaceFlag_Selected;
        }
        DocSelectionChanged( doc );
    }

    void DocTargetFaces( const EditorDoc & doc, List<FaceRef> & outFaces ) {
        ListClear( outFaces );
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            const Brush & brush = doc.map.brushes[i];
            for( i32 f = 0; f < brush.faces.count; f++ ) {
                if( brush.faces[f].flags & BrushFaceFlag_Selected ) {
                    ListAdd( outFaces, FaceRef{ i, f } );
                }
            }
        }
        if( outFaces.count > 0 ) {
            return;
        }
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            if( !DocIsSelected( doc, i ) ) {
                continue;
            }
            for( i32 f = 0; f < doc.map.brushes[i].faces.count; f++ ) {
                ListAdd( outFaces, FaceRef{ i, f } );
            }
        }
    }

    static void GrowBounds( const Brush & brush, bool * any, Vec3 * min, Vec3 * max ) {
        Vec3 brushMin = {};
        Vec3 brushMax = {};
        BrushBounds( brush, &brushMin, &brushMax );
        if( !*any ) {
            *min = brushMin;
            *max = brushMax;
            *any = true;
            return;
        }
        *min = Vec3{ Min( min->x, brushMin.x ), Min( min->y, brushMin.y ), Min( min->z, brushMin.z ) };
        *max = Vec3{ Max( max->x, brushMax.x ), Max( max->y, brushMax.y ), Max( max->z, brushMax.z ) };
    }

    bool DocSelectionBounds( const EditorDoc & doc, Vec3 * outMin, Vec3 * outMax ) {
        bool any = false;
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            if( DocIsSelected( doc, i ) ) {
                GrowBounds( doc.map.brushes[i], &any, outMin, outMax );
            }
        }
        return any;
    }

    bool DocVisibleBounds( const EditorDoc & doc, Vec3 * outMin, Vec3 * outMax ) {
        bool any = false;
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            if( BrushVisible( doc.map.brushes[i] ) ) {
                GrowBounds( doc.map.brushes[i], &any, outMin, outMax );
            }
        }
        return any;
    }

    bool DocPick( const EditorDoc & doc, Vec3 origin, Vec3 direction, i32 * outBrush, i32 * outFace, f32 * outDistance ) {
        i32 bestBrush = kNoBrush;
        i32 bestFace = -1;
        f32 bestDistance = 0.0f;
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            const Brush & brush = doc.map.brushes[i];
            if( !BrushVisible( brush ) ) {
                continue;
            }
            f32 distance = 0.0f;
            i32 face = -1;
            if( BrushRaycast( brush, origin, direction, &distance, &face ) &&
                ( bestBrush == kNoBrush || distance < bestDistance ) ) {
                bestBrush = i;
                bestFace = face;
                bestDistance = distance;
            }
        }
        *outBrush = bestBrush;
        *outFace = bestFace;
        *outDistance = bestDistance;
        return bestBrush != kNoBrush;
    }

    // --- edits --------------------------------------------------------------

    i32 DocAddBrush( EditorDoc & doc, Brush & brush ) {
        if( ListAdd( doc.map.brushes, brush ) == nullptr ) {
            BrushFree( brush );
            return kNoBrush;
        }
        brush = {};
        return doc.map.brushes.count - 1;
    }

    bool DocCopySelected( EditorDoc & doc, Vec3 offset ) {
        const i32 count = doc.map.brushes.count;
        bool any = false;
        for( i32 i = 0; i < count; i++ ) {
            if( !DocIsSelected( doc, i ) ) {
                continue;
            }
            Brush copy = BrushCopy( doc.map.brushes[i] );
            BrushTranslate( copy, offset, true );
            doc.map.brushes[i].flags &= ~BrushFlag_Selected;
            DocAddBrush( doc, copy );
            any = true;
        }
        DocTouch( doc );
        return any;
    }

    bool DocDuplicateSelected( EditorDoc & doc, Vec3 offset ) {
        if( DocSelectedCount( doc ) == 0 ) {
            return false;
        }
        DocBeginEdit( doc );
        DocCopySelected( doc, offset );
        DocEdited( doc );
        return true;
    }

    bool DocDeleteSelected( EditorDoc & doc ) {
        if( DocSelectedCount( doc ) == 0 ) {
            return false;
        }
        DocBeginEdit( doc );
        for( i32 i = doc.map.brushes.count - 1; i >= 0; i-- ) {
            if( DocIsSelected( doc, i ) ) {
                BrushFree( doc.map.brushes[i] );
                ListRemoveIndex( doc.map.brushes, i );
            }
        }
        DocEdited( doc );
        return true;
    }

    bool DocTranslateSelected( EditorDoc & doc, Vec3 delta, bool textureLock ) {
        if( DocSelectedCount( doc ) == 0 ) {
            return false;
        }
        DocBeginEdit( doc );
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            if( DocIsSelected( doc, i ) ) {
                BrushTranslate( doc.map.brushes[i], delta, textureLock );
            }
        }
        DocEdited( doc );
        return true;
    }

    bool DocTransformSelected( EditorDoc & doc, const Mat4 & transform ) {
        if( DocSelectedCount( doc ) == 0 ) {
            return false;
        }
        DocBeginEdit( doc );
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            // Transformed as a copy, so one brush that cannot take the
            // transform aborts the lot rather than leaving half the selection
            // turned.
            if( DocIsSelected( doc, i ) && !BrushTransform( doc.map.brushes[i], transform ) ) {
                DocAbandonEdit( doc );
                return false;
            }
        }
        DocEdited( doc );
        return true;
    }

    // The selected brushes are taken out and the result appended selected, so
    // every CSG operation below has the same shape.
    static void SplitSelection( EditorDoc & doc, List<Brush> & outKept, List<Brush> & outSelected ) {
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            ListAdd( DocIsSelected( doc, i ) ? outSelected : outKept, BrushCopy( doc.map.brushes[i] ) );
        }
    }

    bool DocMergeSelected( EditorDoc & doc ) {
        List<Brush> kept = {};
        List<Brush> selected = {};
        SplitSelection( doc, kept, selected );
        if( selected.count < 2 ) {
            FreeBrushList( kept );
            FreeBrushList( selected );
            return false;
        }

        List<Vec3> points = {};
        List<BrushFace> templates = {};
        for( i32 i = 0; i < selected.count; i++ ) {
            ListAddRange( points, selected[i].points );
            ListAddRange( templates, selected[i].faces );
        }

        Brush merged = {};
        const bool ok = BrushCreateHull( merged, points.data, points.count, templates.data, templates.count, FaceTextureDefault() );
        ListFree( points );
        ListFree( templates );
        FreeBrushList( selected );

        if( !ok ) {
            BrushFree( merged );
            FreeBrushList( kept );
            return false;
        }

        for( i32 f = 0; f < merged.faces.count; f++ ) {
            merged.faces[f].flags = 0;
        }
        merged.flags = BrushFlag_Selected;
        ListAdd( kept, merged );

        DocBeginEdit( doc );
        ReplaceBrushes( doc, kept );
        DocEdited( doc );
        return true;
    }

    bool DocSubtractSelected( EditorDoc & doc ) {
        List<Brush> kept = {};
        List<Brush> cutters = {};
        SplitSelection( doc, kept, cutters );
        if( cutters.count == 0 ) {
            FreeBrushList( kept );
            return false;
        }

        List<Brush> result = {};
        List<Brush> pieces = {};
        List<Brush> next = {};
        bool changed = false;

        for( i32 i = 0; i < kept.count; i++ ) {
            Brush & brush = kept[i];
            bool touched = false;
            for( i32 c = 0; c < cutters.count && !touched; c++ ) {
                touched = BrushVisible( brush ) && BrushIntersects( brush, cutters[c] );
            }
            if( !touched ) {
                ListAdd( result, brush );
                brush = {};
                continue;
            }

            changed = true;
            ListAdd( pieces, brush );
            brush = {};
            for( i32 c = 0; c < cutters.count; c++ ) {
                for( i32 p = 0; p < pieces.count; p++ ) {
                    BrushSubtract( pieces[p], cutters[c], next );
                }
                FreeBrushList( pieces );
                pieces = next;
                next = {};
            }
            for( i32 p = 0; p < pieces.count; p++ ) {
                pieces[p].flags = BrushFlag_Selected;
                ClearFaceFlags( pieces[p] );
                ListAdd( result, pieces[p] );
            }
            ListFree( pieces );
        }

        FreeBrushList( kept );
        FreeBrushList( cutters );

        if( !changed ) {
            FreeBrushList( result );
            return false;
        }

        DocBeginEdit( doc );
        ReplaceBrushes( doc, result );
        DocEdited( doc );
        return true;
    }

    bool DocHollowSelected( EditorDoc & doc, f32 thickness ) {
        List<Brush> kept = {};
        List<Brush> selected = {};
        SplitSelection( doc, kept, selected );
        bool changed = false;

        for( i32 i = 0; i < selected.count; i++ ) {
            Brush inner = BrushCopy( selected[i] );
            for( i32 f = 0; f < inner.faces.count; f++ ) {
                inner.faces[f].plane.distance -= thickness;
            }

            // Too thin to leave a room inside: left as it was rather than
            // turned into a solid lump of wall.
            if( !BrushRebuild( inner ) ) {
                BrushFree( inner );
                ListAdd( kept, selected[i] );
                selected[i] = {};
                continue;
            }

            List<Brush> walls = {};
            BrushSubtract( selected[i], inner, walls );
            for( i32 w = 0; w < walls.count; w++ ) {
                walls[w].flags = BrushFlag_Selected;
                ClearFaceFlags( walls[w] );
                ListAdd( kept, walls[w] );
            }
            ListFree( walls );
            BrushFree( inner );
            changed = true;
        }
        FreeBrushList( selected );

        if( !changed ) {
            FreeBrushList( kept );
            return false;
        }

        DocBeginEdit( doc );
        ReplaceBrushes( doc, kept );
        DocEdited( doc );
        return true;
    }

    bool DocIntersectSelected( EditorDoc & doc ) {
        List<Brush> kept = {};
        List<Brush> selected = {};
        SplitSelection( doc, kept, selected );
        if( selected.count < 2 ) {
            FreeBrushList( kept );
            FreeBrushList( selected );
            return false;
        }

        Brush result = BrushCopy( selected[0] );
        bool ok = true;
        for( i32 i = 1; i < selected.count && ok; i++ ) {
            const Brush & other = selected[i];
            for( i32 f = 0; f < other.faces.count && ok; f++ ) {
                Brush clipped = {};
                ok = BrushClipBehind( result, other.faces[f].plane, other.faces[f].texture, &clipped );
                BrushFree( result );
                result = clipped;
            }
        }
        FreeBrushList( selected );

        if( !ok ) {
            BrushFree( result );
            FreeBrushList( kept );
            return false;
        }

        result.flags = BrushFlag_Selected;
        ClearFaceFlags( result );
        ListAdd( kept, result );

        DocBeginEdit( doc );
        ReplaceBrushes( doc, kept );
        DocEdited( doc );
        return true;
    }

    bool DocClipSelected( EditorDoc & doc, Plane plane, bool keepBack, bool keepFront ) {
        if( DocSelectedCount( doc ) == 0 ) {
            return false;
        }

        List<Brush> result = {};
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            const Brush & brush = doc.map.brushes[i];
            if( !DocIsSelected( doc, i ) ) {
                ListAdd( result, BrushCopy( brush ) );
                continue;
            }

            Brush piece = {};
            if( keepBack && BrushClipBehind( brush, plane, BrushNearestTexture( brush, plane.normal ), &piece ) ) {
                ClearFaceFlags( piece );
                piece.flags = BrushFlag_Selected;
                ListAdd( result, piece );
            }
            const Plane flipped = PlaneFlip( plane );
            if( keepFront && BrushClipBehind( brush, flipped, BrushNearestTexture( brush, flipped.normal ), &piece ) ) {
                ClearFaceFlags( piece );
                piece.flags = BrushFlag_Selected;
                ListAdd( result, piece );
            }
        }

        DocBeginEdit( doc );
        ReplaceBrushes( doc, result );
        DocEdited( doc );
        return true;
    }

    bool DocApplyMaterial( EditorDoc & doc, StringView material ) {
        List<FaceRef> faces = {};
        DocTargetFaces( doc, faces );
        if( faces.count == 0 ) {
            ListFree( faces );
            return false;
        }

        DocBeginEdit( doc );
        for( i32 i = 0; i < faces.count; i++ ) {
            StringSet( doc.map.brushes[faces[i].brush].faces[faces[i].face].texture.material, material );
        }
        ListFree( faces );
        DocEdited( doc );
        return true;
    }

    bool DocHideSelected( EditorDoc & doc ) {
        if( DocSelectedCount( doc ) == 0 ) {
            return false;
        }
        DocBeginEdit( doc );
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            Brush & brush = doc.map.brushes[i];
            if( brush.flags & BrushFlag_Selected ) {
                // A hidden brush cannot be seen or clicked, so it must not
                // stay selected where an edit could still reach it unseen.
                brush.flags = ( brush.flags | BrushFlag_Hidden ) & ~BrushFlag_Selected;
            }
        }
        DocEdited( doc );
        return true;
    }

    bool DocShowAll( EditorDoc & doc ) {
        bool any = false;
        for( i32 i = 0; i < doc.map.brushes.count && !any; i++ ) {
            any = !BrushVisible( doc.map.brushes[i] );
        }
        if( !any ) {
            return false;
        }
        DocBeginEdit( doc );
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            doc.map.brushes[i].flags &= ~BrushFlag_Hidden;
        }
        DocEdited( doc );
        return true;
    }

} // namespace sol
