// sol_editor_tools.cpp : what the left button and the edit keys do in each
// tool. The window side of VulkanView - panes, cameras, the frame - lives in
// sol_editor_view.cpp.
#include "sol_editor_view.h"

#include <cmath>

namespace sol {

    // How far a press has to travel, in logical pixels, before it is a drag.
    // Below this it is a click, however long the button was held.
    constexpr i32 kDragThreshold = 4;
    // How near a brush edge in a 2D pane a press has to land to grab the face
    // behind it.
    constexpr f32 kEdgeGrabRadius = 7.0f;
    constexpr f32 kHandleGrabRadius = 9.0f;
    // A drag point further than this along a grazing ray is the horizon, not
    // somewhere anyone meant to put a brush.
    constexpr f32 kMaxDragDistance = 2000.0f;

    constexpr Vec3 kHoverColor = { 1.0f, 0.86f, 0.25f };
    constexpr Vec3 kClipKeepColor = { 1.0f, 0.95f, 0.85f };
    constexpr Vec3 kClipDropColor = { 1.0f, 0.25f, 0.2f };
    constexpr Vec3 kClipPointColor = { 1.0f, 0.62f, 0.2f };
    constexpr Vec3 kHandleColor = { 0.55f, 0.75f, 1.0f };
    constexpr Vec3 kHandleSelectedColor = { 1.0f, 0.55f, 0.15f };
    constexpr Vec3 kMarqueeColor = { 0.9f, 0.9f, 0.95f };
    constexpr Vec4 kTranslucentTint = { 1.0f, 1.0f, 1.0f, 0.28f };

    static i32 DominantAxis( Vec3 v ) {
        const f32 ax = fabsf( v.x );
        const f32 ay = fabsf( v.y );
        const f32 az = fabsf( v.z );
        if( ax >= ay && ax >= az ) {
            return 0;
        }
        return ay >= az ? 1 : 2;
    }

    static f32 Component( Vec3 v, i32 axis ) {
        return axis == 0 ? v.x : ( axis == 1 ? v.y : v.z );
    }

    static void SetComponent( Vec3 * v, i32 axis, f32 value ) {
        if( axis == 0 )      { v->x = value; }
        else if( axis == 1 ) { v->y = value; }
        else                 { v->z = value; }
    }

    static Vec3 AxisVector( i32 axis ) {
        Vec3 v = {};
        SetComponent( &v, axis, 1.0f );
        return v;
    }

    // The signed world axis v points most along.
    static Vec3 SnapToAxis( Vec3 v ) {
        const i32 axis = DominantAxis( v );
        return AxisVector( axis ) * ( Component( v, axis ) < 0.0f ? -1.0f : 1.0f );
    }

    static bool IsZero( Vec3 v ) {
        return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f;
    }

    static bool Near( Vec3 a, Vec3 b ) {
        return fabsf( a.x - b.x ) <= kBrushEpsilon && fabsf( a.y - b.y ) <= kBrushEpsilon && fabsf( a.z - b.z ) <= kBrushEpsilon;
    }

    static void AddUnique( List<Vec3> & points, Vec3 p ) {
        for( i32 i = 0; i < points.count; i++ ) {
            if( Near( points[i], p ) ) {
                return;
            }
        }
        ListAdd( points, p );
    }

    static Mat4 RotationAbout( Vec3 axis, f32 radians ) {
        const i32 index = DominantAxis( axis );
        const f32 angle = Component( axis, index ) < 0.0f ? -radians : radians;
        return index == 0 ? Mat4RotateX( angle ) : ( index == 1 ? Mat4RotateY( angle ) : Mat4RotateZ( angle ) );
    }

    static f32 SegmentDistance( f32 px, f32 py, f32 ax, f32 ay, f32 bx, f32 by ) {
        const f32 dx = bx - ax;
        const f32 dy = by - ay;
        const f32 lengthSquared = dx * dx + dy * dy;
        f32 t = lengthSquared > 0.0f ? ( ( px - ax ) * dx + ( py - ay ) * dy ) / lengthSquared : 0.0f;
        t = t < 0.0f ? 0.0f : ( t > 1.0f ? 1.0f : t );
        const f32 cx = ax + dx * t - px;
        const f32 cy = ay + dy * t - py;
        return sqrtf( cx * cx + cy * cy );
    }

    // --- press, move, release -----------------------------------------------

    void VulkanView::LeftPress( QPoint position, i32 pane, Qt::KeyboardModifiers modifiers ) {
        const bool ctrl = ( modifiers & Qt::ControlModifier ) != 0;
        const bool shift = ( modifiers & Qt::ShiftModifier ) != 0;
        const bool alt = ( modifiers & Qt::AltModifier ) != 0;

        pressPosition = position;
        pressPane = pane;
        pressModifiers = modifiers;
        pressPick = PickAt( position, pane );
        leftDrag = LeftDrag_Pending;
        pendingDrag = LeftDrag_None;
        hoverTargetCount = 0;

        switch( tool ) {
            case EditorTool_Brush: {
                if( alt ) {
                    pendingDrag = LeftDrag_Marquee;
                    break;
                }

                FaceDragTarget targets[2] = {};
                f32 edgeDistance = 0.0f;
                const i32 count = FindFaceTargets( position, pane, targets, 2, &edgeDistance );
                // Shift says "the face, not the brush": a resize, an extrude
                // with Ctrl, or - when nothing selected is under the cursor - a
                // new brush started straight in height mode.
                if( shift ) {
                    if( count > 0 ) {
                        faceTargets[0] = targets[0];
                        faceTargets[1] = targets[1];
                        faceTargetCount = ctrl ? 1 : count;
                        pendingDrag = ctrl ? LeftDrag_Extrude : LeftDrag_Resize;
                    } else {
                        pendingDrag = LeftDrag_Create;
                    }
                    break;
                }

                // In 2D the edges of the selection are handles in their own
                // right, the way they are in Hammer, so resizing needs no key.
                if( !ctrl && EdgeGrab2D( position, pane, count, edgeDistance ) ) {
                    faceTargets[0] = targets[0];
                    faceTargets[1] = targets[1];
                    faceTargetCount = count;
                    pendingDrag = LeftDrag_Resize;
                    break;
                }

                const bool onSelection = PaneIs2D( pane ) ? SelectedUnderCursor( position, pane )
                                                          : ( pressPick.brush != kNoBrush && DocIsSelected( doc, pressPick.brush ) );
                pendingDrag = onSelection ? LeftDrag_Move : LeftDrag_Create;
                break;
            }

            case EditorTool_Clip: {
                // Grabbing a point already placed moves it rather than adding one.
                f32 best = kHandleGrabRadius;
                clipDragPoint = -1;
                for( i32 i = 0; i < clipPointCount; i++ ) {
                    f32 x = 0.0f;
                    f32 y = 0.0f;
                    if( !WorldToScreen( pane, clipPoints[i], &x, &y ) ) {
                        continue;
                    }
                    const f32 d = sqrtf( ( x - (f32)position.x() ) * ( x - (f32)position.x() ) + ( y - (f32)position.y() ) * ( y - (f32)position.y() ) );
                    if( d < best ) {
                        best = d;
                        clipDragPoint = i;
                    }
                }
                if( clipDragPoint >= 0 ) {
                    pendingDrag = LeftDrag_ClipPoint;
                }
                break;
            }

            case EditorTool_Vertex:
            case EditorTool_Edge:
            case EditorTool_Face: {
                Vec3 handle = {};
                if( HandleAt( position, pane, &handle ) ) {
                    // Ctrl toggles on the click instead; a Ctrl press is never
                    // the start of a drag.
                    if( !ctrl ) {
                        if( !HandleSelected( handle ) ) {
                            ListClear( handleSelection );
                            ListAdd( handleSelection, handle );
                        }
                        handleDragStart = handle;
                        pendingDrag = LeftDrag_Handles;
                    }
                } else {
                    pendingDrag = LeftDrag_Marquee;
                }
                break;
            }

            case EditorTool_Rotate: {
                Vec3 origin = {};
                Vec3 direction = {};
                RayAt( position, pane, &origin, &direction );
                // The rings sit where the last frame drew them, which is where
                // the user is aiming.
                const GizmoAxis axis = DocSelectedCount( doc ) > 0 ? GizmoPick( gizmo, origin, direction ) : GizmoAxis_None;
                if( axis != GizmoAxis_None ) {
                    Transform start = TransformDefault();
                    start.position = gizmo.center;
                    if( GizmoBeginDrag( gizmo, axis, start, origin, direction ) ) {
                        // A ring is unmistakable, so the drag starts on the
                        // press rather than waiting for the threshold.
                        StartDrag( LeftDrag_Rotate, position, modifiers );
                    }
                }
                break;
            }

            default:
                break;
        }
    }

    void VulkanView::LeftMove( QPoint position, Qt::KeyboardModifiers modifiers ) {
        if( leftDrag == LeftDrag_Pending ) {
            const QPoint travelled = position - pressPosition;
            if( travelled.manhattanLength() < kDragThreshold || pendingDrag == LeftDrag_None ) {
                return;
            }
            StartDrag( pendingDrag, position, modifiers );
            if( leftDrag == LeftDrag_Pending || leftDrag == LeftDrag_None ) {
                return;
            }
        }
        UpdateDrag( position, modifiers );
    }

    void VulkanView::LeftRelease( QPoint position, Qt::KeyboardModifiers modifiers ) {
        if( leftDrag == LeftDrag_Pending ) {
            leftDrag = LeftDrag_None;
            pendingDrag = LeftDrag_None;
            // The modifiers of the press, not the release: a click is decided
            // by what was held when it started.
            LeftClick( position, pressPane, pressModifiers );
            return;
        }
        FinishDrag( position, modifiers );
    }

    void VulkanView::ModifiersChanged( Qt::KeyboardModifiers modifiers ) {
        if( leftDrag != LeftDrag_None && leftDrag != LeftDrag_Pending ) {
            UpdateDrag( mapFromGlobal( QCursor::pos() ), modifiers );
        } else if( leftDrag == LeftDrag_None && cameraDrag == CameraDrag_None ) {
            HoverMove( mapFromGlobal( QCursor::pos() ), modifiers );
        }
    }

    void VulkanView::HoverMove( QPoint position, Qt::KeyboardModifiers modifiers ) {
        const i32 pane = PaneAt( position );
        hoverTargetCount = 0;
        hoverPane = pane;
        Qt::CursorShape shape = Qt::ArrowCursor;

        if( tool == EditorTool_Brush ) {
            // Shows which faces a Shift-drag would take before it is made: in
            // 2D always, since there the edges answer without Shift.
            const bool shift = ( modifiers & Qt::ShiftModifier ) != 0;
            if( PaneIs2D( pane ) || shift ) {
                f32 edgeDistance = 0.0f;
                hoverTargetCount = FindFaceTargets( position, pane, hoverTargets, 2, &edgeDistance );
                // Shown only where a press would really take the edge.
                if( !shift && !EdgeGrab2D( position, pane, hoverTargetCount, edgeDistance ) ) {
                    hoverTargetCount = 0;
                }
            }
            if( hoverTargetCount == 1 && PaneIs2D( pane ) ) {
                Vec3 right = {};
                Vec3 up = {};
                Vec3 forward = {};
                PaneAxes( pane, &right, &up, &forward );
                shape = fabsf( Vec3Dot( hoverTargets[0].normal, right ) ) > 0.5f ? Qt::SizeHorCursor : Qt::SizeVerCursor;
            } else if( hoverTargetCount > 0 ) {
                shape = Qt::SizeAllCursor;
            } else if( !( modifiers & Qt::AltModifier ) ) {
                const bool onSelection = PaneIs2D( pane ) ? SelectedUnderCursor( position, pane ) : DocIsSelected( doc, PickAt( position, pane ).brush );
                shape = onSelection ? Qt::SizeAllCursor : Qt::CrossCursor;
            }
        } else if( tool == EditorTool_Vertex || tool == EditorTool_Edge || tool == EditorTool_Face ) {
            Vec3 handle = {};
            shape = HandleAt( position, pane, &handle ) ? Qt::PointingHandCursor : Qt::ArrowCursor;
        } else if( tool == EditorTool_Clip ) {
            shape = Qt::CrossCursor;
        } else if( tool == EditorTool_Rotate ) {
            Vec3 origin = {};
            Vec3 direction = {};
            RayAt( position, pane, &origin, &direction );
            gizmo.hovered = GizmoPick( gizmo, origin, direction );
            shape = gizmo.hovered != GizmoAxis_None ? Qt::PointingHandCursor : Qt::ArrowCursor;
        }

        if( shape != hoverCursor ) {
            hoverCursor = shape;
            setCursor( shape );
        }
    }

    void VulkanView::LeftClick( QPoint position, i32 pane, Qt::KeyboardModifiers modifiers ) {
        const bool ctrl = ( modifiers & Qt::ControlModifier ) != 0;
        const bool shift = ( modifiers & Qt::ShiftModifier ) != 0;
        const bool alt = ( modifiers & Qt::AltModifier ) != 0;
        const PickResult & pick = pressPick;

        switch( tool ) {
            case EditorTool_Brush:
            case EditorTool_Rotate: {
                if( tool == EditorTool_Brush && alt ) {
                    PaintMaterialAt( position, pane, ctrl );
                    return;
                }

                if( tool == EditorTool_Brush && shift ) {
                    // Faces are selected for texturing. Picking one also picks
                    // up its material, so the next brush or paint wears it.
                    if( pick.brush == kNoBrush ) {
                        DocSelectNone( doc );
                    } else if( ctrl ) {
                        const bool selected = ( doc.map.brushes[pick.brush].faces[pick.face].flags & BrushFaceFlag_Selected ) != 0;
                        DocSetFaceSelected( doc, pick.brush, pick.face, !selected );
                    } else {
                        DocSelectNone( doc );
                        DocSetFaceSelected( doc, pick.brush, pick.face, true );
                        // Material, scale and rotation carry over; the offsets
                        // only mean anything where this face is.
                        currentTexture = doc.map.brushes[pick.brush].faces[pick.face].texture;
                        currentTexture.offsetU = 0.0f;
                        currentTexture.offsetV = 0.0f;
                    }
                    AfterSelectionReplaced();
                    return;
                }

                if( ctrl ) {
                    if( pick.brush != kNoBrush ) {
                        DocSetSelected( doc, pick.brush, !DocIsSelected( doc, pick.brush ) );
                    }
                } else {
                    DocSelectOnly( doc, pick.brush );
                }
                AfterSelectionReplaced();
                return;
            }

            case EditorTool_Clip:
                if( clipPointCount >= kMaxClipPoints ) {
                    ShowMessage( QStringLiteral( "Three points already: drag one to move it, or Escape to start over" ) );
                } else if( PlaceClipPoint( position, pane, clipPointCount ) ) {
                    clipPointCount++;
                }
                return;

            case EditorTool_Vertex:
            case EditorTool_Edge:
            case EditorTool_Face: {
                Vec3 handle = {};
                if( HandleAt( position, pane, &handle ) ) {
                    if( ctrl ) {
                        SetHandleSelected( handle, !HandleSelected( handle ) );
                    }
                    return;
                }
                // Clicking another brush makes it the one being reshaped, so
                // the tool never has to be put away to move on.
                if( pick.brush != kNoBrush && !DocIsSelected( doc, pick.brush ) ) {
                    if( ctrl ) {
                        DocSetSelected( doc, pick.brush, true );
                    } else {
                        DocSelectOnly( doc, pick.brush );
                        ListClear( handleSelection );
                    }
                    AfterSelectionReplaced();
                } else if( !ctrl ) {
                    ListClear( handleSelection );
                }
                return;
            }

            default:
                return;
        }
    }

    // --- drags --------------------------------------------------------------

    void VulkanView::StartDrag( LeftDrag kind, QPoint position, Qt::KeyboardModifiers modifiers ) {
        leftDrag = kind;
        dragDelta = {};
        dragChanged = false;
        movePhaseStarted = false;
        moveBase = {};

        switch( kind ) {
            case LeftDrag_Create: {
                createValid = false;
                createHeightMode = false;
                if( !PaneIs2D( pressPane ) ) {
                    Vec3 start = {};
                    if( pressPick.brush != kNoBrush ) {
                        // Rises off the face that was pressed, along the world
                        // axis it most faces. An axis-aligned face gives its
                        // exact plane; a slanted one gives the nearest grid
                        // plane through the press.
                        createAxis = DominantAxis( pressPick.normal );
                        createSign = Component( pressPick.normal, createAxis ) < 0.0f ? -1.0f : 1.0f;
                        const f32 along = Component( pressPick.normal, createAxis );
                        const Plane facePlane = doc.map.brushes[pressPick.brush].faces[pressPick.face].plane;
                        createPlane = fabsf( along ) > 0.9999f ? facePlane.distance * along : SnapTo( Component( pressPick.point, createAxis ), grid.step );
                        start = pressPick.point;
                    } else {
                        Vec3 origin = {};
                        Vec3 direction = {};
                        RayAt( pressPosition, pressPane, &origin, &direction );
                        Vec3 local = {};
                        if( !EditorGridRaycast( grid, origin, direction, &local ) ) {
                            leftDrag = LeftDrag_None;
                            ShowMessage( QStringLiteral( "Nothing to build on there: aim at the grid or a surface" ) );
                            return;
                        }
                        start = EditorGridToWorld( grid, local );
                        if( Vec3Length( start - origin ) > kMaxDragDistance ) {
                            leftDrag = LeftDrag_None;
                            return;
                        }
                        createAxis = DominantAxis( grid.normal );
                        createSign = 1.0f;
                        createPlane = Component( grid.position, createAxis );
                    }

                    for( i32 a = 0; a < 3; a++ ) {
                        SetComponent( &start, a, a == createAxis ? createPlane : SnapTo( Component( start, a ), grid.step ) );
                    }
                    createStart = start;
                    createEnd = start;
                    const f32 height = Max( Component( referenceMax, createAxis ) - Component( referenceMin, createAxis ), grid.step );
                    createHeight = createSign * height;
                }
                UpdateCreate( position, modifiers );
                return;
            }

            case LeftDrag_Move:
                DocBeginEdit( doc );
                // Ctrl leaves the originals behind and drags copies away.
                if( pressModifiers & Qt::ControlModifier ) {
                    DocCopySelected( doc, Vec3{} );
                    dragChanged = true;
                }
                SnapshotSelection();
                UpdateMove( position, modifiers );
                return;

            case LeftDrag_Resize:
                DocBeginEdit( doc );
                SnapshotSelection();
                faceAmounts[0] = 0.0f;
                faceAmounts[1] = 0.0f;
                UpdateFaceDrag( position );
                return;

            case LeftDrag_Extrude:
                extrudeValid = false;
                faceAmounts[0] = 0.0f;
                faceAmounts[1] = 0.0f;
                UpdateFaceDrag( position );
                return;

            case LeftDrag_Handles:
                DocBeginEdit( doc );
                SnapshotSelection();
                SnapshotHandles();
                UpdateDrag( position, modifiers );
                return;

            case LeftDrag_Rotate:
                DocBeginEdit( doc );
                SnapshotSelection();
                rotateAngle = 0.0f;
                return;

            case LeftDrag_Marquee:
            case LeftDrag_ClipPoint:
            default:
                UpdateDrag( position, modifiers );
                return;
        }
    }

    void VulkanView::UpdateDrag( QPoint position, Qt::KeyboardModifiers modifiers ) {
        switch( leftDrag ) {
            case LeftDrag_Create:
                UpdateCreate( position, modifiers );
                break;

            case LeftDrag_Move:
                UpdateMove( position, modifiers );
                break;

            case LeftDrag_Resize:
            case LeftDrag_Extrude:
                UpdateFaceDrag( position );
                break;

            case LeftDrag_Handles: {
                Vec3 delta = {};
                if( PaneIs2D( pressPane ) ) {
                    delta = DragTranslation2D( position, handleDragStart );
                } else if( !DragTranslation3D( position, handleDragStart, ( modifiers & Qt::AltModifier ) != 0, handleDragStart, &delta ) ) {
                    break;
                }
                if( !( delta.x == dragDelta.x && delta.y == dragDelta.y && delta.z == dragDelta.z ) ) {
                    // A delta that would fold a brush is refused and the last
                    // good one stands, so the handle sticks rather than jumps.
                    ApplyHandleDelta( delta );
                }
                break;
            }

            case LeftDrag_ClipPoint:
                if( clipDragPoint >= 0 && clipDragPoint < clipPointCount ) {
                    PlaceClipPoint( position, pressPane, clipDragPoint );
                }
                break;

            case LeftDrag_Rotate: {
                Vec3 origin = {};
                Vec3 direction = {};
                RayAt( position, pressPane, &origin, &direction );
                Transform turned = {};
                const f32 snap = ( modifiers & Qt::ControlModifier ) ? 0.0f : kGizmoRotateSnap;
                if( !GizmoUpdateDrag( gizmo, origin, direction, 0.0f, snap, &turned ) ) {
                    break;
                }
                const i32 axis = (i32)gizmo.active - 1;
                const f32 angle = Component( turned.rotation, axis );
                if( angle == rotateAngle ) {
                    break;
                }

                const Mat4 rotation = axis == 0 ? Mat4RotateX( angle ) : ( axis == 1 ? Mat4RotateY( angle ) : Mat4RotateZ( angle ) );
                const Mat4 transform = Mat4Translate( gizmo.center ) * rotation * Mat4Translate( gizmo.center * -1.0f );

                List<Brush> turnedBrushes = {};
                bool ok = true;
                for( i32 i = 0; i < dragOriginals.count && ok; i++ ) {
                    Brush copy = BrushCopy( dragOriginals[i] );
                    ok = BrushTransform( copy, transform );
                    ListAdd( turnedBrushes, copy );
                }
                for( i32 i = 0; i < turnedBrushes.count; i++ ) {
                    if( ok ) {
                        BrushFree( doc.map.brushes[dragIndices[i]] );
                        doc.map.brushes[dragIndices[i]] = turnedBrushes[i];
                    } else {
                        BrushFree( turnedBrushes[i] );
                    }
                }
                ListFree( turnedBrushes );
                if( ok ) {
                    rotateAngle = angle;
                    DocTouch( doc );
                }
                break;
            }

            default:
                break;
        }
    }

    void VulkanView::FinishDrag( QPoint position, Qt::KeyboardModifiers modifiers ) {
        SPLATS_UNUSED( modifiers );

        switch( leftDrag ) {
            case LeftDrag_Create:
                if( createValid ) {
                    DocBeginEdit( doc );
                    DocSelectNone( doc );
                    Brush brush = createBrush;
                    createBrush = {};
                    brush.flags = BrushFlag_Selected;
                    DocAddBrush( doc, brush );
                    DocEdited( doc );
                    AfterSelectionReplaced();
                }
                createValid = false;
                previewVersion++;
                break;

            case LeftDrag_Move:
                if( dragChanged || !IsZero( dragDelta ) ) {
                    DocEdited( doc );
                } else {
                    DocAbandonEdit( doc );
                }
                break;

            case LeftDrag_Resize:
                if( faceAmounts[0] != 0.0f || faceAmounts[1] != 0.0f ) {
                    DocEdited( doc );
                    Vec3 min = {};
                    Vec3 max = {};
                    if( DocSelectionBounds( doc, &min, &max ) ) {
                        referenceMin = min;
                        referenceMax = max;
                    }
                } else {
                    DocAbandonEdit( doc );
                }
                break;

            case LeftDrag_Extrude:
                if( extrudeValid ) {
                    DocBeginEdit( doc );
                    DocSelectNone( doc );
                    Brush brush = extrudeBrush;
                    extrudeBrush = {};
                    brush.flags = BrushFlag_Selected;
                    DocAddBrush( doc, brush );
                    DocEdited( doc );
                    AfterSelectionReplaced();
                }
                extrudeValid = false;
                previewVersion++;
                break;

            case LeftDrag_Marquee:
                MarqueeSelect( pressPosition, position, pressPane, ( pressModifiers & Qt::ControlModifier ) != 0 );
                break;

            case LeftDrag_Handles:
                if( !IsZero( dragDelta ) ) {
                    DocEdited( doc );
                    for( i32 i = 0; i < handleSelection.count; i++ ) {
                        handleSelection[i] = handleSelection[i] + dragDelta;
                    }
                    PruneHandleSelection();
                } else {
                    DocAbandonEdit( doc );
                }
                break;

            case LeftDrag_Rotate:
                GizmoEndDrag( gizmo );
                if( rotateAngle != 0.0f ) {
                    DocEdited( doc );
                } else {
                    DocAbandonEdit( doc );
                }
                break;

            default:
                break;
        }

        FreeSnapshot();
        leftDrag = LeftDrag_None;
        pendingDrag = LeftDrag_None;
        clipDragPoint = -1;
    }

    void VulkanView::CancelDrag() {
        switch( leftDrag ) {
            case LeftDrag_Move:
            case LeftDrag_Resize:
            case LeftDrag_Handles:
            case LeftDrag_Rotate:
                // Each of these began an edit and has been rewriting the map
                // since; the snapshot it took puts everything back.
                DocAbandonEdit( doc );
                break;
            case LeftDrag_Create:
            case LeftDrag_Extrude:
                previewVersion++;
                break;
            default:
                break;
        }

        GizmoEndDrag( gizmo );
        createValid = false;
        extrudeValid = false;
        FreeSnapshot();
        leftDrag = LeftDrag_None;
        pendingDrag = LeftDrag_None;
        clipDragPoint = -1;
    }

    // --- snapshots ----------------------------------------------------------

    void VulkanView::SnapshotSelection() {
        FreeSnapshot();
        DocSelectedBrushes( doc, dragIndices );
        for( i32 i = 0; i < dragIndices.count; i++ ) {
            ListAdd( dragOriginals, BrushCopy( doc.map.brushes[dragIndices[i]] ) );
        }
        if( !DocSelectionBounds( doc, &dragBoundsMin, &dragBoundsMax ) ) {
            dragBoundsMin = {};
            dragBoundsMax = {};
        }
    }

    void VulkanView::FreeSnapshot() {
        for( i32 i = 0; i < dragOriginals.count; i++ ) {
            BrushFree( dragOriginals[i] );
        }
        ListFree( dragOriginals );
        ListFree( dragIndices );
    }

    void VulkanView::ApplyTranslation( Vec3 delta ) {
        for( i32 i = 0; i < dragIndices.count; i++ ) {
            Brush & target = doc.map.brushes[dragIndices[i]];
            BrushFree( target );
            target = BrushCopy( dragOriginals[i] );
            BrushTranslate( target, delta, textureLock );
        }
        dragDelta = delta;
        DocTouch( doc );
    }

    // --- translation --------------------------------------------------------

    Vec3 VulkanView::SnapAxes( Vec3 raw, Vec3 reference, bool x, bool y, bool z ) const {
        const bool snap[3] = { x, y, z };
        Vec3 result = raw;
        for( i32 a = 0; a < 3; a++ ) {
            if( snap[a] ) {
                const f32 start = Component( reference, a );
                SetComponent( &result, a, SnapTo( start + Component( raw, a ), grid.step ) - start );
            }
        }
        return result;
    }

    Vec3 VulkanView::DragTranslation2D( QPoint position, Vec3 reference ) const {
        Vec3 right = {};
        Vec3 up = {};
        Vec3 forward = {};
        PaneAxes( pressPane, &right, &up, &forward );

        Vec3 raw = PlanePoint2D( position, pressPane ) - PlanePoint2D( pressPosition, pressPane );
        SetComponent( &raw, DominantAxis( forward ), 0.0f );

        const i32 a = DominantAxis( right );
        const i32 b = DominantAxis( up );
        return SnapAxes( raw, reference, a == 0 || b == 0, a == 1 || b == 1, a == 2 || b == 2 );
    }

    bool VulkanView::DragTranslation3D( QPoint position, Vec3 grabbed, bool vertical, Vec3 reference, Vec3 * outDelta ) {
        Vec3 origin = {};
        Vec3 direction = {};
        RayAt( position, pressPane, &origin, &direction );
        const Vec3 upAxis = { 0.0f, 1.0f, 0.0f };

        // A new phase starts from wherever the last one left the grabbed point,
        // measured from wherever the cursor is now, so nothing jumps.
        if( !movePhaseStarted || vertical != moveVertical ) {
            moveBase = dragDelta;
            moveVertical = vertical;
            movePhaseStarted = true;
            moveAnchor = grabbed + moveBase;
            if( vertical ) {
                if( !RayLineClosest( origin, direction, moveAnchor, upAxis, &moveAnchorT ) ) {
                    moveAnchorT = 0.0f;
                }
            } else {
                f32 t = 0.0f;
                movePhaseHit = moveAnchor;
                if( RayPlaneIntersect( origin, direction, moveAnchor, upAxis, &t ) && t > 0.0f && t < kMaxDragDistance ) {
                    movePhaseHit = origin + direction * t;
                }
            }
        }

        Vec3 raw = moveBase;
        if( vertical ) {
            f32 t = 0.0f;
            if( !RayLineClosest( origin, direction, moveAnchor, upAxis, &t ) ) {
                return false;
            }
            raw.y = moveBase.y + ( t - moveAnchorT );
            *outDelta = SnapAxes( raw, reference, false, true, false );
        } else {
            f32 t = 0.0f;
            if( !RayPlaneIntersect( origin, direction, moveAnchor, upAxis, &t ) || t <= 0.0f || t > kMaxDragDistance ) {
                return false;
            }
            const Vec3 hit = origin + direction * t;
            raw.x = moveBase.x + ( hit.x - movePhaseHit.x );
            raw.z = moveBase.z + ( hit.z - movePhaseHit.z );
            *outDelta = SnapAxes( raw, reference, true, false, true );
        }
        return true;
    }

    bool VulkanView::SelectedUnderCursor( QPoint position, i32 pane ) const {
        Vec3 origin = {};
        Vec3 direction = {};
        RayAt( position, pane, &origin, &direction );
        for( i32 i = 0; i < doc.map.brushes.count; i++ ) {
            f32 distance = 0.0f;
            i32 face = -1;
            if( DocIsSelected( doc, i ) && BrushRaycast( doc.map.brushes[i], origin, direction, &distance, &face ) ) {
                return true;
            }
        }
        return false;
    }

    void VulkanView::UpdateMove( QPoint position, Qt::KeyboardModifiers modifiers ) {
        Vec3 delta = {};
        if( PaneIs2D( pressPane ) ) {
            delta = DragTranslation2D( position, dragBoundsMin );
        } else if( !DragTranslation3D( position, pressPick.point, ( modifiers & Qt::AltModifier ) != 0, dragBoundsMin, &delta ) ) {
            return;
        }
        if( !( delta.x == dragDelta.x && delta.y == dragDelta.y && delta.z == dragDelta.z ) ) {
            ApplyTranslation( delta );
        }
    }

    // --- create -------------------------------------------------------------

    void VulkanView::UpdateCreate( QPoint position, Qt::KeyboardModifiers modifiers ) {
        const f32 step = grid.step;
        Vec3 min = {};
        Vec3 max = {};

        if( PaneIs2D( pressPane ) ) {
            Vec3 right = {};
            Vec3 up = {};
            Vec3 forward = {};
            PaneAxes( pressPane, &right, &up, &forward );
            const Vec3 a = PlanePoint2D( pressPosition, pressPane );
            const Vec3 b = PlanePoint2D( position, pressPane );

            const i32 inPlane[2] = { DominantAxis( right ), DominantAxis( up ) };
            for( i32 i = 0; i < 2; i++ ) {
                const i32 axis = inPlane[i];
                const f32 sa = SnapTo( Component( a, axis ), step );
                const f32 sb = SnapTo( Component( b, axis ), step );
                f32 lo = Min( sa, sb );
                f32 hi = Max( sa, sb );
                if( hi - lo < step ) {
                    hi = lo + step;
                }
                SetComponent( &min, axis, lo );
                SetComponent( &max, axis, hi );
            }

            // The depth a 2D pane cannot show comes from the last brush made
            // or selected, so walls drawn in the top view share a height.
            const i32 depthAxis = DominantAxis( forward );
            f32 lo = Component( referenceMin, depthAxis );
            f32 hi = Component( referenceMax, depthAxis );
            if( hi - lo < step ) {
                hi = lo + step;
            }
            SetComponent( &min, depthAxis, lo );
            SetComponent( &max, depthAxis, hi );
        } else {
            Vec3 origin = {};
            Vec3 direction = {};
            RayAt( position, pressPane, &origin, &direction );
            const Vec3 axisVector = AxisVector( createAxis );

            Vec3 baseCenter = ( createStart + createEnd ) * 0.5f;
            SetComponent( &baseCenter, createAxis, createPlane );

            if( modifiers & Qt::ShiftModifier ) {
                // Height mode: the base is locked and the cursor pulls the box
                // up the axis through the middle of it.
                if( !createHeightMode ) {
                    createHeightMode = true;
                    if( !RayLineClosest( origin, direction, baseCenter, axisVector, &createHeightAnchorT ) ) {
                        createHeightAnchorT = 0.0f;
                    }
                    createHeightAtAnchor = createHeight;
                }
                f32 t = 0.0f;
                if( RayLineClosest( origin, direction, baseCenter, axisVector, &t ) ) {
                    f32 height = SnapTo( createHeightAtAnchor + ( t - createHeightAnchorT ), step );
                    if( fabsf( height ) < step ) {
                        const bool below = height < 0.0f || ( height == 0.0f && createHeight < 0.0f );
                        height = below ? -step : step;
                    }
                    createHeight = height;
                }
            } else {
                createHeightMode = false;
                f32 t = 0.0f;
                if( RayPlaneIntersect( origin, direction, axisVector * createPlane, axisVector, &t ) && t > 0.0f && t < kMaxDragDistance ) {
                    Vec3 hit = origin + direction * t;
                    for( i32 a = 0; a < 3; a++ ) {
                        SetComponent( &hit, a, a == createAxis ? createPlane : SnapTo( Component( hit, a ), step ) );
                    }
                    createEnd = hit;
                }
            }

            for( i32 a = 0; a < 3; a++ ) {
                if( a == createAxis ) {
                    SetComponent( &min, a, Min( createPlane, createPlane + createHeight ) );
                    SetComponent( &max, a, Max( createPlane, createPlane + createHeight ) );
                    continue;
                }
                f32 lo = Min( Component( createStart, a ), Component( createEnd, a ) );
                f32 hi = Max( Component( createStart, a ), Component( createEnd, a ) );
                if( hi - lo < step ) {
                    hi = lo + step;
                }
                SetComponent( &min, a, lo );
                SetComponent( &max, a, hi );
            }
        }

        FaceTexture texture = currentTexture;
        createValid = BrushCreateBox( createBrush, min, max, texture );
        previewVersion++;
    }

    // --- resize and extrude ---------------------------------------------------

    f32 VulkanView::SnapFaceAmount( Vec3 normal, f32 startDistance, f32 raw ) const {
        // An axis-aligned face lands its plane on the grid; a slanted one can
        // only be moved in whole steps, since its plane never meets the grid.
        if( fabsf( Component( normal, DominantAxis( normal ) ) ) > 0.9999f ) {
            return SnapTo( startDistance + raw, grid.step ) - startDistance;
        }
        return SnapTo( raw, grid.step );
    }

    bool VulkanView::EdgeGrab2D( QPoint position, i32 pane, i32 targetCount, f32 edgeDistance ) const {
        if( !PaneIs2D( pane ) || targetCount == 0 ) {
            return false;
        }
        // Inside a brush only the rim right against the edge resizes; the rest
        // of it moves. Otherwise a brush a few grid cells across on screen
        // would be all edge and could never be dragged.
        return edgeDistance <= 3.0f || !SelectedUnderCursor( position, pane );
    }

    i32 VulkanView::FindFaceTargets( QPoint position, i32 pane, FaceDragTarget * outTargets, i32 maxTargets, f32 * outEdgeDistance ) const {
        if( outEdgeDistance != nullptr ) {
            *outEdgeDistance = 0.0f;
        }
        if( maxTargets <= 0 ) {
            return 0;
        }

        if( !PaneIs2D( pane ) ) {
            const PickResult pick = PickAt( position, pane );
            if( pick.brush == kNoBrush || !DocIsSelected( doc, pick.brush ) ) {
                return 0;
            }
            const Plane plane = doc.map.brushes[pick.brush].faces[pick.face].plane;
            outTargets[0] = FaceDragTarget{ pick.brush, pick.face, plane.normal, plane.distance };
            return 1;
        }

        // In 2D a face seen edge on is a line, and a press near that line
        // grabs it. Near a corner two lines qualify and both are taken.
        Vec3 right = {};
        Vec3 up = {};
        Vec3 forward = {};
        PaneAxes( pane, &right, &up, &forward );
        const Vec3 cursorWorld = PlanePoint2D( position, pane );

        struct Candidate {
            FaceDragTarget  target;
            f32             score;
            f32             distance;
        };
        List<Candidate> candidates = {};

        for( i32 b = 0; b < doc.map.brushes.count; b++ ) {
            if( !DocIsSelected( doc, b ) ) {
                continue;
            }
            const Brush & brush = doc.map.brushes[b];
            for( i32 f = 0; f < brush.faces.count; f++ ) {
                const BrushFace & face = brush.faces[f];
                if( fabsf( Vec3Dot( face.plane.normal, forward ) ) > 1e-3f ) {
                    continue;
                }

                // The polygon projects to a segment: its two furthest corners.
                f32 ax = 0.0f;
                f32 ay = 0.0f;
                f32 bx = 0.0f;
                f32 by = 0.0f;
                f32 longest = -1.0f;
                for( i32 i = 0; i < face.pointCount; i++ ) {
                    f32 xi = 0.0f;
                    f32 yi = 0.0f;
                    WorldToScreen( pane, brush.points[face.firstPoint + i], &xi, &yi );
                    for( i32 j = i + 1; j < face.pointCount; j++ ) {
                        f32 xj = 0.0f;
                        f32 yj = 0.0f;
                        WorldToScreen( pane, brush.points[face.firstPoint + j], &xj, &yj );
                        const f32 length = ( xj - xi ) * ( xj - xi ) + ( yj - yi ) * ( yj - yi );
                        if( length > longest ) {
                            longest = length;
                            ax = xi;
                            ay = yi;
                            bx = xj;
                            by = yj;
                        }
                    }
                }

                const f32 distance = SegmentDistance( (f32)position.x(), (f32)position.y(), ax, ay, bx, by );
                if( distance > kEdgeGrabRadius ) {
                    continue;
                }
                // Between a face and the one back to back with it - a thin
                // brush, or two brushes touching - the one the cursor is
                // outside of wins. The face is edge on, so the cursor's depth
                // does not change which side of it the cursor is on.
                const bool outside = PlaneSide( face.plane, cursorWorld ) > 0.0f;
                const Candidate candidate = { FaceDragTarget{ b, f, face.plane.normal, face.plane.distance }, distance - ( outside ? 2.0f : 0.0f ), distance };
                ListAdd( candidates, candidate );
            }
        }

        // The nearest edge, then - near a corner - the nearest edge turned
        // another way, so both sides of the corner move together.
        i32 count = 0;
        i32 first = -1;
        for( i32 i = 0; i < candidates.count; i++ ) {
            if( first < 0 || candidates[i].score < candidates[first].score ) {
                first = i;
            }
        }
        if( first >= 0 ) {
            outTargets[count++] = candidates[first].target;
            if( outEdgeDistance != nullptr ) {
                *outEdgeDistance = candidates[first].distance;
            }
            i32 second = -1;
            for( i32 i = 0; i < candidates.count && maxTargets > 1; i++ ) {
                const bool turned = fabsf( Vec3Dot( candidates[i].target.normal, candidates[first].target.normal ) ) < 0.99f;
                if( turned && ( second < 0 || candidates[i].score < candidates[second].score ) ) {
                    second = i;
                }
            }
            if( second >= 0 ) {
                outTargets[count++] = candidates[second].target;
            }
        }
        ListFree( candidates );
        return count;
    }

    bool VulkanView::ApplyFaceMove( const f32 * amounts ) {
        List<Brush> moved = {};
        bool ok = true;
        for( i32 i = 0; i < dragOriginals.count && ok; i++ ) {
            const Brush & original = dragOriginals[i];
            Brush copy = BrushCopy( original );
            bool touched = false;
            // Every selected brush with a face turned the same way as a
            // grabbed one moves that face too, so a row of walls is raised by
            // dragging the top of any of them.
            for( i32 k = 0; k < faceTargetCount; k++ ) {
                for( i32 f = 0; f < copy.faces.count; f++ ) {
                    if( Vec3Dot( copy.faces[f].plane.normal, faceTargets[k].normal ) > 1.0f - 1e-4f ) {
                        copy.faces[f].plane.distance += amounts[k];
                        touched = true;
                    }
                }
            }
            if( touched && ( !BrushRebuild( copy ) || copy.faces.count != original.faces.count ) ) {
                ok = false;
            }
            ListAdd( moved, copy );
        }

        for( i32 i = 0; i < moved.count; i++ ) {
            if( ok ) {
                BrushFree( doc.map.brushes[dragIndices[i]] );
                doc.map.brushes[dragIndices[i]] = moved[i];
            } else {
                BrushFree( moved[i] );
            }
        }
        ListFree( moved );
        if( ok ) {
            DocTouch( doc );
        }
        return ok;
    }

    void VulkanView::UpdateFaceDrag( QPoint position ) {
        f32 amounts[2] = { 0.0f, 0.0f };
        for( i32 k = 0; k < faceTargetCount; k++ ) {
            const FaceDragTarget & target = faceTargets[k];
            f32 raw = 0.0f;
            if( PaneIs2D( pressPane ) ) {
                raw = Vec3Dot( PlanePoint2D( position, pressPane ) - PlanePoint2D( pressPosition, pressPane ), target.normal );
            } else {
                Vec3 origin = {};
                Vec3 direction = {};
                RayAt( position, pressPane, &origin, &direction );
                if( !RayLineClosest( origin, direction, pressPick.point, target.normal, &raw ) ) {
                    return;
                }
            }
            amounts[k] = SnapFaceAmount( target.normal, target.startDistance, raw );
        }

        if( leftDrag == LeftDrag_Resize ) {
            if( ( amounts[0] != faceAmounts[0] || amounts[1] != faceAmounts[1] ) && ApplyFaceMove( amounts ) ) {
                faceAmounts[0] = amounts[0];
                faceAmounts[1] = amounts[1];
            }
            return;
        }

        // Extrude: a new brush grows out of the face; the source is untouched.
        if( amounts[0] == faceAmounts[0] && extrudeValid ) {
            return;
        }
        faceAmounts[0] = amounts[0];
        const FaceDragTarget & target = faceTargets[0];
        extrudeValid = amounts[0] > kBrushEpsilon && target.brush >= 0 && target.brush < doc.map.brushes.count &&
                       BrushCreateExtrusion( extrudeBrush, doc.map.brushes[target.brush], target.face, amounts[0] );
        previewVersion++;
    }

    // --- vertex, edge and face handles -----------------------------------------

    void VulkanView::BuildHandles( List<Vec3> & outHandles ) const {
        ListClear( outHandles );
        List<Vec3> scratch = {};
        for( i32 b = 0; b < doc.map.brushes.count; b++ ) {
            if( !DocIsSelected( doc, b ) ) {
                continue;
            }
            const Brush & brush = doc.map.brushes[b];
            if( tool == EditorTool_Vertex ) {
                BrushVertices( brush, scratch );
                for( i32 i = 0; i < scratch.count; i++ ) {
                    AddUnique( outHandles, scratch[i] );
                }
            } else if( tool == EditorTool_Edge ) {
                BrushEdges( brush, scratch );
                for( i32 i = 0; i + 1 < scratch.count; i += 2 ) {
                    AddUnique( outHandles, ( scratch[i] + scratch[i + 1] ) * 0.5f );
                }
            } else if( tool == EditorTool_Face ) {
                for( i32 f = 0; f < brush.faces.count; f++ ) {
                    AddUnique( outHandles, BrushFaceCenter( brush, f ) );
                }
            }
        }
        ListFree( scratch );
    }

    bool VulkanView::HandleAt( QPoint position, i32 pane, Vec3 * outHandle ) const {
        List<Vec3> handles = {};
        BuildHandles( handles );

        Vec3 origin = {};
        Vec3 direction = {};
        RayAt( position, pane, &origin, &direction );

        bool found = false;
        f32 bestDistance = kHandleGrabRadius;
        f32 bestDepth = 0.0f;
        for( i32 i = 0; i < handles.count; i++ ) {
            f32 x = 0.0f;
            f32 y = 0.0f;
            if( !WorldToScreen( pane, handles[i], &x, &y ) ) {
                continue;
            }
            const f32 dx = x - (f32)position.x();
            const f32 dy = y - (f32)position.y();
            const f32 distance = sqrtf( dx * dx + dy * dy );
            const f32 depth = Vec3Dot( handles[i] - origin, direction );
            // Handles stacked on screen - every corner of a box seen square on
            // - tie on distance, and the one nearest the camera wins.
            const bool closer = distance < bestDistance - 0.5f;
            const bool tiedAndNearer = found && distance < bestDistance + 0.5f && depth < bestDepth;
            if( distance <= kHandleGrabRadius && ( !found || closer || tiedAndNearer ) ) {
                found = true;
                bestDistance = distance;
                bestDepth = depth;
                *outHandle = handles[i];
            }
        }
        ListFree( handles );
        return found;
    }

    bool VulkanView::HandleSelected( Vec3 handle ) const {
        for( i32 i = 0; i < handleSelection.count; i++ ) {
            if( Near( handleSelection[i], handle ) ) {
                return true;
            }
        }
        return false;
    }

    void VulkanView::SetHandleSelected( Vec3 handle, bool selected ) {
        for( i32 i = 0; i < handleSelection.count; i++ ) {
            if( Near( handleSelection[i], handle ) ) {
                if( !selected ) {
                    ListRemoveIndex( handleSelection, i );
                }
                return;
            }
        }
        if( selected ) {
            ListAdd( handleSelection, handle );
        }
    }

    void VulkanView::PruneHandleSelection() {
        if( handleSelection.count == 0 ) {
            return;
        }
        List<Vec3> handles = {};
        BuildHandles( handles );
        for( i32 i = handleSelection.count - 1; i >= 0; i-- ) {
            bool present = false;
            for( i32 h = 0; h < handles.count && !present; h++ ) {
                present = Near( handles[h], handleSelection[i] );
            }
            if( !present ) {
                ListRemoveIndex( handleSelection, i );
            }
        }
        ListFree( handles );
    }

    void VulkanView::SnapshotHandles() {
        ListClear( handleVertexStart );
        ListClear( handleVertices );
        ListClear( handleAffected );

        List<Vec3> vertices = {};
        List<Vec3> edges = {};
        for( i32 i = 0; i < dragOriginals.count; i++ ) {
            const Brush & brush = dragOriginals[i];
            BrushVertices( brush, vertices );
            BrushEdges( brush, edges );
            ListAdd( handleVertexStart, handleVertices.count );

            for( i32 v = 0; v < vertices.count; v++ ) {
                const Vec3 p = vertices[v];
                bool affected = false;
                if( tool == EditorTool_Vertex ) {
                    affected = HandleSelected( p );
                } else if( tool == EditorTool_Edge ) {
                    for( i32 e = 0; e + 1 < edges.count && !affected; e += 2 ) {
                        affected = ( Near( edges[e], p ) || Near( edges[e + 1], p ) ) && HandleSelected( ( edges[e] + edges[e + 1] ) * 0.5f );
                    }
                } else if( tool == EditorTool_Face ) {
                    for( i32 f = 0; f < brush.faces.count && !affected; f++ ) {
                        const BrushFace & face = brush.faces[f];
                        bool onFace = false;
                        for( i32 k = 0; k < face.pointCount && !onFace; k++ ) {
                            onFace = Near( brush.points[face.firstPoint + k], p );
                        }
                        affected = onFace && HandleSelected( BrushFaceCenter( brush, f ) );
                    }
                }
                ListAdd( handleVertices, p );
                ListAdd( handleAffected, (u8)( affected ? 1 : 0 ) );
            }
        }
        ListAdd( handleVertexStart, handleVertices.count );
        ListFree( vertices );
        ListFree( edges );
    }

    bool VulkanView::ApplyHandleDelta( Vec3 delta ) {
        List<Brush> reshaped = {};
        List<Vec3> points = {};
        bool ok = true;

        for( i32 i = 0; i < dragOriginals.count && ok; i++ ) {
            const Brush & original = dragOriginals[i];
            const i32 first = handleVertexStart[i];
            const i32 last = handleVertexStart[i + 1];

            ListClear( points );
            bool any = false;
            for( i32 v = first; v < last; v++ ) {
                const bool affected = handleAffected[v] != 0;
                ListAdd( points, affected ? handleVertices[v] + delta : handleVertices[v] );
                any = any || affected;
            }

            if( !any ) {
                ListAdd( reshaped, BrushCopy( original ) );
                continue;
            }

            // The corners are the edit and the planes are rebuilt round them,
            // which keeps the brush convex whatever the drag does. Each new
            // face keeps the material of the face it most resembles.
            Brush hull = {};
            ok = BrushCreateHull( hull, points.data, points.count, original.faces.data, original.faces.count, currentTexture );
            hull.flags = original.flags;
            ListAdd( reshaped, hull );
        }

        for( i32 i = 0; i < reshaped.count; i++ ) {
            if( ok ) {
                BrushFree( doc.map.brushes[dragIndices[i]] );
                doc.map.brushes[dragIndices[i]] = reshaped[i];
            } else {
                BrushFree( reshaped[i] );
            }
        }
        ListFree( reshaped );
        ListFree( points );

        if( ok ) {
            dragDelta = delta;
            DocTouch( doc );
        }
        return ok;
    }

    void VulkanView::NudgeHandles( Vec3 delta ) {
        DocBeginEdit( doc );
        SnapshotSelection();
        SnapshotHandles();
        dragDelta = {};
        if( ApplyHandleDelta( delta ) ) {
            DocEdited( doc );
            for( i32 i = 0; i < handleSelection.count; i++ ) {
                handleSelection[i] = handleSelection[i] + delta;
            }
            PruneHandleSelection();
        } else {
            DocAbandonEdit( doc );
            ShowMessage( QStringLiteral( "That would flatten a brush" ) );
        }
        FreeSnapshot();
        dragDelta = {};
    }

    void VulkanView::DeleteSelectedVertices() {
        DocBeginEdit( doc );
        SnapshotSelection();

        List<Vec3> vertices = {};
        List<Vec3> kept = {};
        bool any = false;
        bool refused = false;
        for( i32 i = 0; i < dragOriginals.count; i++ ) {
            const Brush & original = dragOriginals[i];
            BrushVertices( original, vertices );
            ListClear( kept );
            for( i32 v = 0; v < vertices.count; v++ ) {
                if( !HandleSelected( vertices[v] ) ) {
                    ListAdd( kept, vertices[v] );
                }
            }
            if( kept.count == vertices.count ) {
                continue;
            }
            Brush hull = {};
            if( BrushCreateHull( hull, kept.data, kept.count, original.faces.data, original.faces.count, currentTexture ) ) {
                hull.flags = original.flags;
                BrushFree( doc.map.brushes[dragIndices[i]] );
                doc.map.brushes[dragIndices[i]] = hull;
                any = true;
            } else {
                BrushFree( hull );
                refused = true;
            }
        }
        ListFree( vertices );
        ListFree( kept );
        FreeSnapshot();

        if( any ) {
            DocEdited( doc );
            ListClear( handleSelection );
        } else {
            DocAbandonEdit( doc );
        }
        if( refused ) {
            ShowMessage( QStringLiteral( "Some brushes kept their corners: removing them would leave nothing solid" ) );
        }
    }

    // --- clip ---------------------------------------------------------------

    bool VulkanView::PlaceClipPoint( QPoint position, i32 pane, i32 index ) {
        if( index < 0 || index >= kMaxClipPoints ) {
            return false;
        }

        if( PaneIs2D( pane ) ) {
            Vec3 right = {};
            Vec3 up = {};
            Vec3 forward = {};
            PaneAxes( pane, &right, &up, &forward );

            // Two points in one 2D pane already fix the plane - it runs along
            // the view - and a third there would put all three in a line along it.
            if( index == 2 && Vec3Length( clipPointAxis[0] ) > 0.5f && Vec3Length( clipPointAxis[1] ) > 0.5f &&
                fabsf( Vec3Dot( clipPointAxis[0], forward ) ) > 0.99f && fabsf( Vec3Dot( clipPointAxis[1], forward ) ) > 0.99f ) {
                ShowMessage( QStringLiteral( "Two points in this view already define the cut: press Enter, or add a third in another view" ) );
                return false;
            }

            Vec3 point = PlanePoint2D( position, pane );
            const i32 depthAxis = DominantAxis( forward );
            Vec3 min = {};
            Vec3 max = {};
            const f32 depth = DocSelectionBounds( doc, &min, &max ) ? Component( ( min + max ) * 0.5f, depthAxis ) : 0.0f;
            for( i32 a = 0; a < 3; a++ ) {
                SetComponent( &point, a, a == depthAxis ? depth : SnapTo( Component( point, a ), grid.step ) );
            }
            clipPoints[index] = point;
            clipPointAxis[index] = SnapToAxis( forward );
            clipPointNormal[index] = {};
            return true;
        }

        const PickResult pick = PickAt( position, pane );
        if( pick.brush == kNoBrush ) {
            return false;
        }
        Vec3 point = Vec3SnapTo( pick.point, grid.step );
        // Snapping all three axes could lift the point off the face; on an
        // axis-aligned face the plane coordinate is put back exactly.
        const i32 axis = DominantAxis( pick.normal );
        if( fabsf( Component( pick.normal, axis ) ) > 0.9999f ) {
            SetComponent( &point, axis, Component( pick.point, axis ) );
        }
        clipPoints[index] = point;
        clipPointAxis[index] = {};
        clipPointNormal[index] = pick.normal;
        return true;
    }

    bool VulkanView::ClipPlane( Plane * outPlane ) const {
        if( clipPointCount < 2 ) {
            return false;
        }

        if( clipPointCount >= 3 && PlaneFromPoints( clipPoints[0], clipPoints[1], clipPoints[2], outPlane ) ) {
            return true;
        }

        // Two points and a direction the plane also runs along: the view axis
        // of a 2D pane they were placed in, or else the normal of the surface
        // the first was placed on, which cuts square to that surface.
        Vec3 along = clipPointAxis[0];
        if( Vec3Length( along ) < 0.5f ) {
            along = clipPointAxis[1];
        }
        if( Vec3Length( along ) < 0.5f ) {
            along = clipPointNormal[0];
        }
        const Vec3 normal = Vec3Cross( clipPoints[1] - clipPoints[0], along );
        if( Vec3Length( normal ) < 1e-6f ) {
            return false;
        }
        *outPlane = PlaneMake( normal, clipPoints[0] );
        return true;
    }

    void VulkanView::ApplyClip() {
        Plane plane = {};
        if( !ClipPlane( &plane ) ) {
            ShowMessage( QStringLiteral( "Place two points in a 2D view, or three on surfaces, first" ) );
            return;
        }
        if( DocSelectedCount( doc ) == 0 ) {
            ShowMessage( QStringLiteral( "Select the brushes to clip first" ) );
            return;
        }
        DocClipSelected( doc, plane, clipSide != ClipSide_Front, clipSide != ClipSide_Back );
        clipPointCount = 0;
        AfterSelectionReplaced();
    }

    // --- selection helpers ----------------------------------------------------

    void VulkanView::MarqueeSelect( QPoint from, QPoint to, i32 pane, bool additive ) {
        const QRect rect = QRect( from, to ).normalized();

        if( tool == EditorTool_Brush ) {
            if( !additive ) {
                DocSelectNone( doc );
            }
            // A brush is taken only when all of it is inside the box, which is
            // what makes a box round one room not also take the floor under it.
            for( i32 b = 0; b < doc.map.brushes.count; b++ ) {
                const Brush & brush = doc.map.brushes[b];
                if( brush.flags & BrushFlag_Hidden ) {
                    continue;
                }
                bool inside = brush.points.count > 0;
                for( i32 p = 0; p < brush.points.count && inside; p++ ) {
                    f32 x = 0.0f;
                    f32 y = 0.0f;
                    inside = WorldToScreen( pane, brush.points[p], &x, &y ) && rect.contains( QPoint( (int)x, (int)y ) );
                }
                if( inside ) {
                    DocSetSelected( doc, b, true );
                }
            }
            AfterSelectionReplaced();
            return;
        }

        if( !additive ) {
            ListClear( handleSelection );
        }
        List<Vec3> handles = {};
        BuildHandles( handles );
        for( i32 i = 0; i < handles.count; i++ ) {
            f32 x = 0.0f;
            f32 y = 0.0f;
            if( WorldToScreen( pane, handles[i], &x, &y ) && rect.contains( QPoint( (int)x, (int)y ) ) ) {
                SetHandleSelected( handles[i], true );
            }
        }
        ListFree( handles );
    }

    void VulkanView::PaintMaterialAt( QPoint position, i32 pane, bool wholeBrush ) {
        const PickResult pick = PickAt( position, pane );
        if( pick.brush == kNoBrush ) {
            return;
        }
        DocBeginEdit( doc );
        Brush & brush = doc.map.brushes[pick.brush];
        for( i32 f = 0; f < brush.faces.count; f++ ) {
            if( wholeBrush || f == pick.face ) {
                brush.faces[f].texture.material = currentTexture.material;
            }
        }
        DocEdited( doc );
    }

    void VulkanView::AfterSelectionReplaced() {
        duplicateActive = false;
        PruneHandleSelection();
        Vec3 min = {};
        Vec3 max = {};
        if( DocSelectionBounds( doc, &min, &max ) ) {
            referenceMin = min;
            referenceMax = max;
        }
    }

    // --- keyboard edits -------------------------------------------------------

    Vec3 VulkanView::PaneWorldAxis( i32 pane, f32 screenX, f32 screenY, f32 depth ) const {
        Vec3 right = {};
        Vec3 up = {};
        Vec3 forward = {};
        PaneAxes( pane, &right, &up, &forward );

        if( PaneIs2D( pane ) ) {
            // Depth is towards the viewer, so it runs against the view.
            return right * screenX + up * screenY - forward * depth;
        }

        // The perspective pane's arrows move across the floor, relative to
        // where the camera faces, snapped to the world axis it faces most.
        if( depth != 0.0f ) {
            return Vec3{ 0.0f, depth, 0.0f };
        }
        const Vec3 flatRight = SnapToAxis( Vec3{ right.x, 0.0f, right.z } );
        const Vec3 flatForward = SnapToAxis( Vec3{ forward.x, 0.0f, forward.z } );
        return flatRight * screenX + flatForward * screenY;
    }

    void VulkanView::Nudge( i32 screenX, i32 screenY, i32 depth ) {
        if( leftDrag != LeftDrag_None ) {
            return;
        }
        const Vec3 delta = PaneWorldAxis( activePane, (f32)screenX, (f32)screenY, (f32)depth ) * grid.step;
        if( ( tool == EditorTool_Vertex || tool == EditorTool_Edge || tool == EditorTool_Face ) && handleSelection.count > 0 ) {
            NudgeHandles( delta );
            return;
        }
        DocTranslateSelected( doc, delta, textureLock );
    }

    void VulkanView::RotateSelection90( bool counterClockwise, bool aboutRight ) {
        CancelDrag();
        Vec3 min = {};
        Vec3 max = {};
        if( !DocSelectionBounds( doc, &min, &max ) ) {
            return;
        }

        Vec3 right = {};
        Vec3 up = {};
        Vec3 forward = {};
        PaneAxes( activePane, &right, &up, &forward );
        Vec3 axis = {};
        if( PaneIs2D( activePane ) ) {
            // Counter-clockwise as seen on screen is about the axis pointing
            // out of it, at the viewer.
            axis = aboutRight ? right : forward * -1.0f;
        } else {
            axis = aboutRight ? SnapToAxis( Vec3{ right.x, 0.0f, right.z } ) : Vec3{ 0.0f, 1.0f, 0.0f };
        }

        // About the grid point nearest the middle, so a brush that sat on the
        // grid stays on it wherever its proportions allow.
        const Vec3 center = Vec3SnapTo( ( min + max ) * 0.5f, grid.step );
        const f32 angle = counterClockwise ? kHalfPi : -kHalfPi;
        const Mat4 transform = Mat4Translate( center ) * RotationAbout( axis, angle ) * Mat4Translate( center * -1.0f );
        DocTransformSelected( doc, transform );
    }

    void VulkanView::FlipSelection( bool vertical ) {
        Vec3 min = {};
        Vec3 max = {};
        if( !DocSelectionBounds( doc, &min, &max ) ) {
            return;
        }

        Vec3 right = {};
        Vec3 up = {};
        Vec3 forward = {};
        PaneAxes( activePane, &right, &up, &forward );
        Vec3 axis = {};
        if( PaneIs2D( activePane ) ) {
            axis = vertical ? up : right;
        } else {
            axis = vertical ? Vec3{ 0.0f, 1.0f, 0.0f } : SnapToAxis( Vec3{ right.x, 0.0f, right.z } );
        }

        // Mirrored about the middle, so the selection keeps its footprint.
        Vec3 scale = { 1.0f, 1.0f, 1.0f };
        SetComponent( &scale, DominantAxis( axis ), -1.0f );
        const Vec3 center = ( min + max ) * 0.5f;
        const Mat4 transform = Mat4Translate( center ) * Mat4Scale( scale ) * Mat4Translate( center * -1.0f );
        DocTransformSelected( doc, transform );
    }

    void VulkanView::DuplicateSelection() {
        Vec3 min = {};
        Vec3 max = {};
        if( !DocSelectionBounds( doc, &min, &max ) ) {
            return;
        }

        Vec3 stride = {};
        if( duplicateActive ) {
            // How far the last copy ended up from its source is the spacing
            // the user chose, whether they dragged it there or left it.
            stride = min - duplicateSourceMin;
            if( IsZero( stride ) ) {
                stride = duplicateStride;
            }
        } else {
            // A first copy lands beside its source, butted up against it along
            // the pane's screen right.
            const Vec3 axis = PaneWorldAxis( activePane, 1.0f, 0.0f, 0.0f );
            const f32 extent = fabsf( Vec3Dot( max - min, axis ) );
            stride = axis * Max( extent, grid.step );
        }

        duplicateSourceMin = min;
        if( DocDuplicateSelected( doc, stride ) ) {
            duplicateStride = stride;
            duplicateActive = true;
            PruneHandleSelection();
        }
    }

    // --- overlay ------------------------------------------------------------

    void VulkanView::DrawToolOverlay( StreamBuilder & builder ) {
        const qreal dpr = devicePixelRatio();

        // Faces a resize or extrude would take, or is taking.
        const FaceDragTarget * targets = nullptr;
        i32 targetCount = 0;
        if( leftDrag == LeftDrag_Resize || leftDrag == LeftDrag_Extrude ) {
            targets = faceTargets;
            targetCount = faceTargetCount;
        } else if( leftDrag == LeftDrag_None && tool == EditorTool_Brush ) {
            targets = hoverTargets;
            targetCount = hoverTargetCount;
        }
        if( targetCount > 0 ) {
            StreamBatch( builder, RenderBatch_Translucent, kTranslucentTint, kRenderAllViews );
            for( i32 k = 0; k < targetCount; k++ ) {
                if( targets[k].brush >= 0 && targets[k].brush < doc.map.brushes.count ) {
                    EditorDrawFaceFill( builder, doc.map.brushes[targets[k].brush], targets[k].face, kHoverColor );
                }
            }
            StreamBatch( builder, RenderBatch_LinesOnTop, kNoTint, kRenderAllViews );
            for( i32 k = 0; k < targetCount; k++ ) {
                if( targets[k].brush < 0 || targets[k].brush >= doc.map.brushes.count ) {
                    continue;
                }
                const Brush & brush = doc.map.brushes[targets[k].brush];
                if( targets[k].face < 0 || targets[k].face >= brush.faces.count ) {
                    continue;
                }
                const BrushFace & face = brush.faces[targets[k].face];
                for( i32 i = 0; i < face.pointCount; i++ ) {
                    StreamLine( builder, brush.points[face.firstPoint + i], brush.points[face.firstPoint + ( i + 1 ) % face.pointCount], kHoverColor );
                }
            }
        }

        if( leftDrag == LeftDrag_Marquee && PaneVisible( pressPane ) ) {
            const QRect rect = PaneRect( pressPane );
            RenderBatch * batch = StreamBatch( builder, RenderBatch_LinesOnTop, kNoTint, 1u << (u32)pressPane );
            batch->screenSpace = true;
            const f32 x0 = (f32)( pressPosition.x() - rect.x() ) + 0.5f;
            const f32 y0 = (f32)( pressPosition.y() - rect.y() ) + 0.5f;
            const f32 x1 = (f32)( lastMouse.x() - rect.x() ) + 0.5f;
            const f32 y1 = (f32)( lastMouse.y() - rect.y() ) + 0.5f;
            EditorDrawScreenRect( builder, x0, y0, x1, y1, (f32)rect.width(), (f32)rect.height(), kMarqueeColor );
        }

        if( tool == EditorTool_Clip ) {
            Plane plane = {};
            if( ClipPlane( &plane ) ) {
                // What goes is filled red; what stays is outlined.
                List<Brush> kept = {};
                List<Brush> dropped = {};
                for( i32 b = 0; b < doc.map.brushes.count; b++ ) {
                    if( !DocIsSelected( doc, b ) ) {
                        continue;
                    }
                    const Brush & brush = doc.map.brushes[b];
                    Brush back = {};
                    Brush front = {};
                    if( BrushClipBehind( brush, plane, FaceTextureDefault(), &back ) ) {
                        ListAdd( clipSide == ClipSide_Front ? dropped : kept, back );
                    }
                    if( BrushClipBehind( brush, PlaneFlip( plane ), FaceTextureDefault(), &front ) ) {
                        ListAdd( clipSide == ClipSide_Back ? dropped : kept, front );
                    }
                }
                StreamBatch( builder, RenderBatch_Translucent, Vec4{ 1.0f, 1.0f, 1.0f, 0.35f }, kRenderAllViews );
                for( i32 i = 0; i < dropped.count; i++ ) {
                    EditorDrawBrushFill( builder, dropped[i], kClipDropColor );
                }
                StreamBatch( builder, RenderBatch_LinesOnTop, kNoTint, kRenderAllViews );
                for( i32 i = 0; i < kept.count; i++ ) {
                    EditorDrawBrushEdges( builder, kept[i], kClipKeepColor );
                }
                for( i32 i = 0; i < kept.count; i++ ) {
                    BrushFree( kept[i] );
                }
                for( i32 i = 0; i < dropped.count; i++ ) {
                    BrushFree( dropped[i] );
                }
                ListFree( kept );
                ListFree( dropped );
            }

            StreamBatch( builder, RenderBatch_LinesOnTop, kNoTint, kRenderAllViews );
            for( i32 i = 0; i + 1 < clipPointCount; i++ ) {
                StreamLine( builder, clipPoints[i], clipPoints[i + 1], kClipPointColor );
            }
            if( clipPointCount == 3 ) {
                StreamLine( builder, clipPoints[2], clipPoints[0], kClipPointColor );
            }
            StreamBatch( builder, RenderBatch_PointsOnTop, kNoTint, kRenderAllViews )->pointSize = (f32)( 10.0 * dpr );
            for( i32 i = 0; i < clipPointCount; i++ ) {
                StreamPoint( builder, clipPoints[i], kClipPointColor );
            }
        }

        if( tool == EditorTool_Vertex || tool == EditorTool_Edge || tool == EditorTool_Face ) {
            List<Vec3> handles = {};
            BuildHandles( handles );

            // Selected edges and faces are shown as the edge or face, not only
            // as the dot in their middle.
            if( tool == EditorTool_Edge || tool == EditorTool_Face ) {
                List<Vec3> edges = {};
                StreamBatch( builder, tool == EditorTool_Face ? RenderBatch_Translucent : RenderBatch_LinesOnTop,
                             tool == EditorTool_Face ? kTranslucentTint : kNoTint, kRenderAllViews );
                for( i32 b = 0; b < doc.map.brushes.count; b++ ) {
                    if( !DocIsSelected( doc, b ) ) {
                        continue;
                    }
                    const Brush & brush = doc.map.brushes[b];
                    if( tool == EditorTool_Edge ) {
                        BrushEdges( brush, edges );
                        for( i32 e = 0; e + 1 < edges.count; e += 2 ) {
                            if( HandleSelected( ( edges[e] + edges[e + 1] ) * 0.5f ) ) {
                                StreamLine( builder, edges[e], edges[e + 1], kHandleSelectedColor );
                            }
                        }
                    } else {
                        for( i32 f = 0; f < brush.faces.count; f++ ) {
                            if( HandleSelected( BrushFaceCenter( brush, f ) ) ) {
                                EditorDrawFaceFill( builder, brush, f, kHandleSelectedColor );
                            }
                        }
                    }
                }
                ListFree( edges );
            }

            StreamBatch( builder, RenderBatch_PointsOnTop, kNoTint, kRenderAllViews )->pointSize = (f32)( 7.0 * dpr );
            for( i32 i = 0; i < handles.count; i++ ) {
                if( !HandleSelected( handles[i] ) ) {
                    StreamPoint( builder, handles[i], kHandleColor );
                }
            }
            StreamBatch( builder, RenderBatch_PointsOnTop, kNoTint, kRenderAllViews )->pointSize = (f32)( 9.0 * dpr );
            for( i32 i = 0; i < handles.count; i++ ) {
                if( HandleSelected( handles[i] ) ) {
                    StreamPoint( builder, handles[i], kHandleSelectedColor );
                }
            }
            ListFree( handles );
        }
    }

} // namespace sol
