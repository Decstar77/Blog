// sol_editor_doc.h : the map being edited, its history, and every edit that
// works on the selection as a whole.
#pragma once

#include "sol_map.h"

namespace sol {

    constexpr i32 kNoBrush = -1;

    // How deep undo goes. Each step is a copy of the whole map, which for a
    // brush map is a few hundred bytes a brush, so this bounds memory at a
    // size a level of a few thousand brushes can afford.
    constexpr i32 kDocMaxUndo = 128;

    struct EditorDoc {
        Map         map;
        // Whole-map snapshots, oldest first. Copying everything per edit is
        // cheaper to get right than recording an inverse for every operation,
        // and because selection lives on the brushes, undo puts the selection
        // back the way it was too.
        List<Map>   undo;
        List<Map>   redo;
        // Bumped by anything that changes how the map looks, selection
        // included. Whoever draws the map rebuilds when this moves.
        u32         version;
        // Unsaved work. Selection alone never sets it.
        bool        modified;
        // What modified was when the open edit began, so abandoning an edit
        // that changed nothing does not leave the document looking dirty.
        bool        modifiedBeforeEdit;
    };

    // A face addressed by brush and face index. Good until the next edit that
    // adds or removes brushes or faces.
    struct FaceRef {
        i32     brush;
        i32     face;
    };

    EditorDoc   DocCreate();
    void        DocFree( EditorDoc & doc );
    // Takes map over as the whole document and drops the old one's history.
    void        DocReplaceMap( EditorDoc & doc, Map & map );

    // --- history ------------------------------------------------------------
    // An edit is bracketed: DocBeginEdit snapshots the map as it is about to
    // change, then either DocEdited keeps the change or DocAbandonEdit puts the
    // snapshot back. A drag begins once, rewrites the map on every move, and
    // settles it on release, which makes the whole drag one undo step.
    void        DocBeginEdit( EditorDoc & doc );
    void        DocEdited( EditorDoc & doc );
    void        DocAbandonEdit( EditorDoc & doc );
    // Something changed that the next frame should draw, but the edit is not
    // settled yet - the middle of a drag.
    void        DocTouch( EditorDoc & doc );
    void        DocSelectionChanged( EditorDoc & doc );
    bool        DocUndo( EditorDoc & doc );
    bool        DocRedo( EditorDoc & doc );

    // --- selection ----------------------------------------------------------
    // Brushes and faces are selected exclusively: selecting either clears the
    // other, since every edit works on one or the other and never both.
    bool        DocIsSelected( const EditorDoc & doc, i32 brush );
    i32         DocSelectedCount( const EditorDoc & doc );
    void        DocSelectedBrushes( const EditorDoc & doc, List<i32> & outIndices );
    void        DocSetSelected( EditorDoc & doc, i32 brush, bool selected );
    // kNoBrush clears the selection.
    void        DocSelectOnly( EditorDoc & doc, i32 brush );
    void        DocSelectAll( EditorDoc & doc );
    // Brushes and faces both.
    void        DocSelectNone( EditorDoc & doc );
    i32         DocSelectedFaceCount( const EditorDoc & doc );
    void        DocSetFaceSelected( EditorDoc & doc, i32 brush, i32 face, bool selected );
    // The faces an edit that works on faces should touch: the selected ones,
    // or every face of every selected brush when no face is selected.
    void        DocTargetFaces( const EditorDoc & doc, List<FaceRef> & outFaces );
    bool        DocSelectionBounds( const EditorDoc & doc, Vec3 * outMin, Vec3 * outMax );
    // Bounds of everything visible, for framing a view on the whole map.
    bool        DocVisibleBounds( const EditorDoc & doc, Vec3 * outMin, Vec3 * outMax );

    // Nearest visible brush along the ray.
    bool        DocPick( const EditorDoc & doc, Vec3 origin, Vec3 direction, i32 * outBrush, i32 * outFace, f32 * outDistance );

    // --- edits --------------------------------------------------------------
    // Everything below records its own undo step and returns whether it
    // changed anything, except where it says it does not.

    // No history: for a caller already inside an edit. Takes the brush over.
    i32         DocAddBrush( EditorDoc & doc, Brush & brush );
    // No history. Copies the selection, moves the copies by offset, and makes
    // them the selection; the originals stay where they were.
    bool        DocCopySelected( EditorDoc & doc, Vec3 offset );

    bool        DocDuplicateSelected( EditorDoc & doc, Vec3 offset );
    bool        DocDeleteSelected( EditorDoc & doc );
    bool        DocTranslateSelected( EditorDoc & doc, Vec3 delta, bool textureLock );
    // Rigid transforms and mirrors only, as BrushTransform.
    bool        DocTransformSelected( EditorDoc & doc, const Mat4 & transform );
    // The selection becomes its convex hull.
    bool        DocMergeSelected( EditorDoc & doc );
    // Carves the selection out of every visible brush it overlaps, then
    // deletes it. The fragments left behind become the selection.
    bool        DocSubtractSelected( EditorDoc & doc );
    // Each selected brush becomes walls of the given thickness around the
    // space it used to fill.
    bool        DocHollowSelected( EditorDoc & doc, f32 thickness );
    // The selection becomes the one brush they all share.
    bool        DocIntersectSelected( EditorDoc & doc );
    // Cuts every selected brush with plane, keeping the part behind it, the
    // part in front, or both as separate brushes. A brush wholly on a side not
    // kept goes. The new faces take the texture of the face they most resemble.
    bool        DocClipSelected( EditorDoc & doc, Plane plane, bool keepBack, bool keepFront );
    bool        DocApplyMaterial( EditorDoc & doc, StringView material );
    bool        DocHideSelected( EditorDoc & doc );
    bool        DocShowAll( EditorDoc & doc );

} // namespace sol
