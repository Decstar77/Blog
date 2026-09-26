#pragma once

#include "sol_camera.h"
#include "sol_editor_doc.h"
#include "sol_editor_draw.h"
#include "sol_editor_gizmo.h"
#include "sol_editor_grid.h"
#include "sol_render.h"

#include <QByteArray>
#include <QElapsedTimer>
#include <QPoint>
#include <QRect>
#include <QString>
#include <QWindow>

class QWidget;

namespace sol {

    enum PaneKind {
        PaneKind_Perspective,
        PaneKind_Ortho,
    };

    struct EditorPane {
        PaneKind            kind;
        // Fraction of the window. A pane the layout does not show has zero
        // size, which the renderer skips and PaneAt never lands in, so the
        // slot index of a pane never changes and view masks can be fixed.
        f32                 x;
        f32                 y;
        f32                 width;
        f32                 height;

        FlyCamera           fly;
        FlyCameraInput      flyInput;
        OrthoCamera         ortho;
        OrthoCameraInput    orthoInput;
    };

    enum PaneLayout {
        PaneLayout_Single,  // perspective only
        PaneLayout_Split,   // perspective and top, side by side
        PaneLayout_Quad,    // perspective, top, front, side
        PaneLayout_Tall,    // a large perspective pane with the three 2D panes stacked beside it
        PaneLayout_Count,
    };

    // What the left button does. One is always up; Escape returns to Brush.
    enum EditorTool {
        EditorTool_Brush,   // select, create, move, resize, extrude
        EditorTool_Clip,    // cut the selection with a plane through two or three points
        EditorTool_Vertex,  // reshape the selection by its corners...
        EditorTool_Edge,    // ...its edges...
        EditorTool_Face,    // ...or its faces
        EditorTool_Rotate,  // turn the selection with rings
        EditorTool_Count,
    };

    // Everything the menus can ask for, so a menu item and its key are one
    // code path.
    enum EditorCommand {
        EditorCommand_New,
        EditorCommand_Open,
        EditorCommand_Save,
        EditorCommand_SaveAs,
        EditorCommand_Undo,
        EditorCommand_Redo,
        EditorCommand_Copy,
        EditorCommand_Cut,
        EditorCommand_Paste,
        EditorCommand_Duplicate,
        EditorCommand_Delete,
        EditorCommand_SelectAll,
        EditorCommand_SelectNone,
        EditorCommand_Hide,
        EditorCommand_ShowAll,
        EditorCommand_Merge,
        EditorCommand_Subtract,
        EditorCommand_Hollow,
        EditorCommand_Intersect,
        EditorCommand_RotateLeft,
        EditorCommand_RotateRight,
        EditorCommand_FlipHorizontal,
        EditorCommand_FlipVertical,
        EditorCommand_ToolBrush,
        EditorCommand_ToolClip,
        EditorCommand_ToolVertex,
        EditorCommand_ToolEdge,
        EditorCommand_ToolFace,
        EditorCommand_ToolRotate,
        EditorCommand_ClipToggleSide,
        EditorCommand_ClipApply,
        EditorCommand_GridFiner,
        EditorCommand_GridCoarser,
        EditorCommand_FrameSelection,
        EditorCommand_ToggleTextureLock,
        EditorCommand_LayoutSingle,
        EditorCommand_LayoutSplit,
        EditorCommand_LayoutQuad,
        EditorCommand_LayoutTall,
        EditorCommand_MaximizePane,
        EditorCommand_Count,
    };

    enum FaceTextureField {
        FaceTextureField_OffsetU,
        FaceTextureField_OffsetV,
        FaceTextureField_ScaleU,
        FaceTextureField_ScaleV,
        FaceTextureField_Rotation,
    };

    enum LeftDrag {
        LeftDrag_None,
        LeftDrag_Pending,   // pressed, not yet moved far enough to be a drag
        LeftDrag_Create,
        LeftDrag_Move,
        LeftDrag_Resize,
        LeftDrag_Extrude,
        LeftDrag_Marquee,
        LeftDrag_Handles,
        LeftDrag_ClipPoint,
        LeftDrag_Rotate,
    };

    enum CameraDrag {
        CameraDrag_None,
        CameraDrag_Look,    // right button in the perspective pane: mouselook plus WASD
        CameraDrag_Orbit,   // Alt + right button in the perspective pane
        CameraDrag_Pan,     // right button in a 2D pane, middle button anywhere
    };

    enum ClipSide {
        ClipSide_Back,
        ClipSide_Front,
        ClipSide_Both,
    };

    // What is under a pixel.
    struct PickResult {
        i32     brush;      // kNoBrush for nothing
        i32     face;
        Vec3    point;
        Vec3    normal;
    };

    // A face grabbed by a resize or extrude, and where its plane started.
    struct FaceDragTarget {
        i32     brush;
        i32     face;
        Vec3    normal;
        f32     startDistance;
    };

    constexpr i32 kMaxClipPoints = 3;

    class VulkanView : public QWindow {
    public:
        explicit VulkanView( Renderer * renderer );
        ~VulkanView() override;

        bool startupFailed() const { return startFailed; }

        // Where file dialogs hang from; the view is a bare QWindow and cannot
        // parent a widget itself.
        void SetDialogParent( QWidget * parent ) { dialogParent = parent; }

        void SetLayout( PaneLayout next );
        PaneLayout CurrentLayout() const { return layout; }

        void Command( EditorCommand command );
        EditorTool CurrentTool() const { return tool; }

        // One line for the status bar: tool, grid, selection and whatever the
        // current drag is measuring.
        QString StatusText() const;
        // What the current tool's buttons do, for the other end of the bar.
        QString ToolHint() const;
        QString DocumentTitle() const;
        bool IsModified() const { return doc.modified; }
        // Asks about unsaved work. False means the user cancelled.
        bool ConfirmDiscard();
        u32 DocVersion() const { return doc.version; }

        // Materials. Setting one makes it what new brushes wear, and applies it
        // to the selection when asked - which is what clicking one in the
        // asset browser does.
        void SetCurrentMaterial( const QString & material, bool applyToSelection );
        QString CurrentMaterial() const;

        // The face inspector. Summary reports the first target face and how
        // many there are; false when nothing is selected.
        bool FaceTextureSummary( FaceTexture * outTexture, i32 * outCount, bool * outMixedMaterial ) const;
        void SetFaceTextureField( FaceTextureField field, f32 value );
        void ResetFaceTextures();
        void FitFaceTextures();

        bool TextureLock() const { return textureLock; }
        f32 GridStep() const { return grid.step; }

    protected:
        void exposeEvent( QExposeEvent * event ) override;
        void resizeEvent( QResizeEvent * event ) override;
        void keyPressEvent( QKeyEvent * event ) override;
        void keyReleaseEvent( QKeyEvent * event ) override;
        void mousePressEvent( QMouseEvent * event ) override;
        void mouseReleaseEvent( QMouseEvent * event ) override;
        void mouseMoveEvent( QMouseEvent * event ) override;
        void wheelEvent( QWheelEvent * event ) override;
        void focusOutEvent( QFocusEvent * event ) override;
        bool event( QEvent * event ) override;

    private:
        // --- sol_editor_view.cpp: window, panes, cameras, frame, input -------
        bool EnsureStarted();
        void Render();
        void BuildBackground();
        void BuildOverlay();
        void UpdateRotateGizmo();

        bool    PaneVisible( i32 pane ) const;
        i32     PaneAt( QPoint position ) const;
        QRect   PaneRect( i32 pane ) const;
        bool    PaneIs2D( i32 pane ) const;
        void    PaneAxes( i32 pane, Vec3 * outRight, Vec3 * outUp, Vec3 * outForward ) const;
        Mat4    PaneViewProjection( i32 pane ) const;
        void    RayAt( QPoint position, i32 pane, Vec3 * outOrigin, Vec3 * outDirection ) const;
        // Window coordinates of a world point, false when it is behind the camera.
        bool    WorldToScreen( i32 pane, Vec3 world, f32 * outX, f32 * outY ) const;
        // The point on a 2D pane's centre plane under a pixel.
        Vec3    PlanePoint2D( QPoint position, i32 pane ) const;
        PickResult PickAt( QPoint position, i32 pane ) const;

        void    BeginCameraDrag( CameraDrag kind, i32 pane, QPoint position );
        void    EndCameraDrag();
        void    SetMovementKey( int key, bool pressed );
        void    OrbitBy( f32 dx, f32 dy );
        void    PanPerspectiveBy( f32 dx, f32 dy );
        void    DollyAt( QPoint position, i32 pane, f32 notches );
        void    ZoomOrthoAt( QPoint position, i32 pane, f32 notches );
        void    FrameBounds( Vec3 min, Vec3 max );

        bool    HandleKey( QKeyEvent * event );
        void    SetGridStepIndex( i32 index );
        void    SetTool( EditorTool next );
        void    ShowMessage( const QString & message );

        bool    FileNew();
        bool    FileOpen();
        bool    FileSave();
        bool    FileSaveAs();
        void    CreateStarterMap();
        // The clipboard carries brushes as map text, so they paste into
        // another map, or another editor, exactly where they were copied from.
        bool    CopySelection();
        void    PasteClipboard();

        // --- sol_editor_tools.cpp: what the left button and the edit keys do -
        void    LeftPress( QPoint position, i32 pane, Qt::KeyboardModifiers modifiers );
        void    LeftMove( QPoint position, Qt::KeyboardModifiers modifiers );
        void    LeftRelease( QPoint position, Qt::KeyboardModifiers modifiers );
        void    HoverMove( QPoint position, Qt::KeyboardModifiers modifiers );
        void    LeftClick( QPoint position, i32 pane, Qt::KeyboardModifiers modifiers );
        // A modifier went down or up with the mouse still: a drag re-reads it
        // (Shift raises a new brush, Alt drags vertically) and hover updates.
        void    ModifiersChanged( Qt::KeyboardModifiers modifiers );
        void    StartDrag( LeftDrag kind, QPoint position, Qt::KeyboardModifiers modifiers );
        void    UpdateDrag( QPoint position, Qt::KeyboardModifiers modifiers );
        void    FinishDrag( QPoint position, Qt::KeyboardModifiers modifiers );
        // Puts everything the drag changed back and forgets it.
        void    CancelDrag();

        void    SnapshotSelection();
        void    FreeSnapshot();
        // Rewrites the dragged brushes as their drag-start copies moved by
        // delta. Every update starts from the snapshot, so a drag that comes
        // back to where it began leaves them exactly as they were.
        void    ApplyTranslation( Vec3 delta );

        // Moves raw so that reference + raw lands on the grid, on the axes
        // asked for. Snapping the result rather than the distance is what puts
        // something that started off the grid back on it.
        Vec3    SnapAxes( Vec3 raw, Vec3 reference, bool x, bool y, bool z ) const;
        // The snapped translation a 2D drag has made so far, in the pane's
        // plane only.
        Vec3    DragTranslation2D( QPoint position, Vec3 reference ) const;
        // The same in 3D, where there is no plane to drag in: across the
        // horizontal plane through what was grabbed, or up and down the
        // vertical line through it while Alt is held. Switching mid-drag keeps
        // what was dragged so far. False when the cursor has left the plane.
        bool    DragTranslation3D( QPoint position, Vec3 grabbed, bool vertical, Vec3 reference, Vec3 * outDelta );
        bool    SelectedUnderCursor( QPoint position, i32 pane ) const;
        f32     SnapFaceAmount( Vec3 normal, f32 startDistance, f32 raw ) const;
        // outEdgeDistance, when asked for, is how far in pixels the cursor is
        // from the first target's edge in a 2D pane.
        i32     FindFaceTargets( QPoint position, i32 pane, FaceDragTarget * outTargets, i32 maxTargets, f32 * outEdgeDistance ) const;
        // Whether a plain press here grabs an edge of the selection in a 2D
        // pane rather than the selection itself.
        bool    EdgeGrab2D( QPoint position, i32 pane, i32 targetCount, f32 edgeDistance ) const;
        bool    ApplyFaceMove( const f32 * amounts );

        void    UpdateCreate( QPoint position, Qt::KeyboardModifiers modifiers );
        void    UpdateMove( QPoint position, Qt::KeyboardModifiers modifiers );
        void    UpdateFaceDrag( QPoint position );

        void    BuildHandles( List<Vec3> & outHandles ) const;
        bool    HandleAt( QPoint position, i32 pane, Vec3 * outHandle ) const;
        bool    HandleSelected( Vec3 handle ) const;
        void    SetHandleSelected( Vec3 handle, bool selected );
        void    PruneHandleSelection();
        void    SnapshotHandles();
        bool    ApplyHandleDelta( Vec3 delta );
        void    NudgeHandles( Vec3 delta );
        void    DeleteSelectedVertices();

        bool    ClipPlane( Plane * outPlane ) const;
        bool    PlaceClipPoint( QPoint position, i32 pane, i32 index );
        void    ApplyClip();

        void    MarqueeSelect( QPoint from, QPoint to, i32 pane, bool additive );
        void    PaintMaterialAt( QPoint position, i32 pane, bool wholeBrush );

        // Arrow keys: one grid step along the pane's own axes.
        void    Nudge( i32 screenX, i32 screenY, i32 depth );
        void    RotateSelection90( bool counterClockwise, bool aboutRight );
        void    FlipSelection( bool vertical );
        void    DuplicateSelection();
        // The world axis a direction in a pane is closest to, signed.
        Vec3    PaneWorldAxis( i32 pane, f32 screenX, f32 screenY, f32 depth ) const;
        void    AfterSelectionReplaced();

        void    DrawToolOverlay( StreamBuilder & builder );

        // --- state ------------------------------------------------------------
        Renderer *          renderer;
        bool                started;
        bool                startFailed;
        QWidget *           dialogParent;

        EditorDoc           doc;
        EditorTextures      textures;
        StreamBuilder       worldStream;
        StreamBuilder       backgroundStream;
        StreamBuilder       overlayStream;
        u32                 builtWorldVersion;
        // Bumped whenever a preview brush changes shape, which redraws the
        // world stream without the document having changed.
        u32                 previewVersion;
        u32                 builtPreviewVersion;

        QString             filePath;
        // The last brushes copied, as map text, for when the system clipboard
        // cannot be read back.
        QByteArray          lastCopied;
        QString             message;
        QElapsedTimer       messageTimer;

        EditorPane          panes[kMaxRenderViews];
        PaneLayout          layout;
        i32                 maximizedPane;
        i32                 activePane;
        // Keys held for the fly camera. Only spent while the right button is
        // looking, so the letters are free for tools the rest of the time.
        FlyCameraInput      movement;

        CameraDrag          cameraDrag;
        // The button that started it, and so the one whose release ends it.
        Qt::MouseButton     cameraButton;
        i32                 cameraPane;
        QPoint              cameraAnchor;
        Vec3                orbitPivot;

        EditorGrid          grid;
        i32                 gridStepIndex;
        bool                textureLock;

        EditorTool          tool;
        // What new brushes wear, and what Alt+click paints.
        FaceTexture         currentTexture;

        // The left button, from press to release.
        LeftDrag            leftDrag;
        LeftDrag            pendingDrag;
        QPoint              pressPosition;
        i32                 pressPane;
        Qt::KeyboardModifiers pressModifiers;
        PickResult          pressPick;
        QPoint              lastMouse;

        // The brushes a drag reshapes, as they were when it started. Indices
        // stay good for the drag: nothing adds or removes brushes mid-drag.
        List<i32>           dragIndices;
        List<Brush>         dragOriginals;
        Vec3                dragBoundsMin;
        Vec3                dragBoundsMax;
        Vec3                dragDelta;
        bool                dragChanged;

        // Move: the part of the delta settled before the current constraint
        // (horizontal or, with Alt, vertical) was taken up, and where the
        // cursor was on the constraint when it was.
        Vec3                moveBase;
        Vec3                moveAnchor;
        Vec3                movePhaseHit;
        f32                 moveAnchorT;
        bool                moveVertical;
        bool                movePhaseStarted;

        // Create: a box grown off an axis plane. In 3D it rises off whatever
        // was under the press, or the grid; in 2D it spans the reference depth.
        Brush               createBrush;
        bool                createValid;
        i32                 createAxis;
        f32                 createSign;
        f32                 createPlane;
        Vec3                createStart;
        Vec3                createEnd;
        f32                 createHeight;
        bool                createHeightMode;
        f32                 createHeightAnchorT;
        f32                 createHeightAtAnchor;
        // Bounds of the last brush made or selected. New brushes take their
        // depth from it, the way TrenchBroom does, so a row of walls comes out
        // the same height without anyone setting it.
        Vec3                referenceMin;
        Vec3                referenceMax;

        // Resize and extrude.
        FaceDragTarget      faceTargets[2];
        i32                 faceTargetCount;
        f32                 faceAmounts[2];
        Brush               extrudeBrush;
        bool                extrudeValid;

        // Hover feedback for the brush tool: the faces a Shift-drag would take.
        FaceDragTarget      hoverTargets[2];
        i32                 hoverTargetCount;
        i32                 hoverPane;
        Qt::CursorShape     hoverCursor;

        // Vertex, edge and face tools. Handles are addressed by position, not
        // index: a reshape rebuilds the brush and renumbers everything, and a
        // corner shared by two selected brushes is one handle for both.
        List<Vec3>          handleSelection;
        Vec3                handleDragStart;
        List<i32>           handleVertexStart;
        List<Vec3>          handleVertices;
        List<u8>            handleAffected;

        Vec3                clipPoints[kMaxClipPoints];
        // The direction a point placed in a 2D pane extends along - the pane's
        // view axis - or zero for a point placed on a surface in 3D.
        Vec3                clipPointAxis[kMaxClipPoints];
        Vec3                clipPointNormal[kMaxClipPoints];
        i32                 clipPointCount;
        i32                 clipDragPoint;
        ClipSide            clipSide;

        Gizmo               gizmo;
        f32                 rotateAngle;

        // Ctrl+D remembers how far the last copy ended up from its source and
        // repeats it, so duplicate, drag into place, duplicate, duplicate lays
        // out a row at the spacing the first drag chose.
        bool                duplicateActive;
        Vec3                duplicateStride;
        Vec3                duplicateSourceMin;

        // Coalesces inspector edits of one field into one undo step.
        i32                 lastFieldEdit;
        u32                 lastFieldEditVersion;

        QElapsedTimer       frameTimer;
    };

    const char * EditorToolName( EditorTool tool );

} // namespace sol
