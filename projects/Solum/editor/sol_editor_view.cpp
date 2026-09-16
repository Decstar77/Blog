#include "sol_editor_view.h"

#include <QCursor>
#include <QFocusEvent>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QStyleHints>
#include <QVulkanInstance>
#include <QWheelEvent>

#include <cmath>
#include <cstdio>

namespace sol {

    // Fraction of the surface the perspective pane gets; the top-down pane
    // takes the rest.
    constexpr f32 kSplitFraction = 0.5f;

    // Selectable grid sizes, smallest first, bound to the 1-6 keys. The snap
    // step follows whichever is current, so the grid is not decoration - it is
    // the thing geometry lands on.
    constexpr f32 kGridSteps[] = { 0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f };
    constexpr i32 kGridStepCount = (i32)( sizeof( kGridSteps ) / sizeof( kGridSteps[0] ) );

    // How far, in logical pixels, a click may land from a vertex and still pick it.
    constexpr f32 kVertexPickRadius = 10.0f;

    VulkanView::VulkanView( Renderer * renderer )
        : renderer( renderer ), world( WorldCreate() ), started( false ), startFailed( false ),
          camera( FlyCameraDefault() ), topCamera( OrthoCameraDefault( OrthoAxis_Top ) ),
          input(), topInput(), dragging( false ), dragPane( Pane_Perspective ),
          createPrimitive( kNoPrimitive ), createStart(),
          createPending( false ), createPressPosition(),
          editPrimitive( kNoPrimitive ), editVertex( kHMNone ), editGeometryDirty( false ),
          gizmo( GizmoCreate() ),
          dragAnchor(), frameTimer() {
        setSurfaceType( QSurface::VulkanSurface );
    }

    VulkanView::~VulkanView() {
        // Authoring data first: it is plain heap memory, and the GPU meshes it
        // was built into belong to the renderer being shut down below.
        WorldFree( world );

        // Runs before the QWindow base destructor, so Qt's surface is still
        // alive while the swapchain that references it is torn down.
        RendererShutdownDevice( renderer );
    }

    bool VulkanView::EnsureStarted() {
        if( started ) {
            return true;
        }
        if( startFailed ) {
            return false;
        }

        QVulkanInstance * qvi = vulkanInstance();
        if( qvi == nullptr ) {
            return false;
        }

        VkSurfaceKHR surface = QVulkanInstance::surfaceForWindow( this );
        if( surface == VK_NULL_HANDLE ) {
            return false;
        }

        const qreal dpr = devicePixelRatio();
        const i32 pixelWidth = (i32)( width() * dpr );
        const i32 pixelHeight = (i32)( height() * dpr );

        // Qt owns the surface, hence ownsSurface = false.
        if( !RendererStartup( renderer, surface, false, pixelWidth, pixelHeight ) ) {
            fprintf( stderr, "Failed to start the renderer on the Qt surface\n" );
            startFailed = true;
            return false;
        }

        WorldCreateDefaultLevel( world, renderer );

        // Local-space geometry that never changes, so it is uploaded once and
        // then only ever repositioned by a push constant.
        List<StaticMeshVertex> gizmoVertices = {};
        GizmoBuildGeometry( gizmo, gizmoVertices );
        RendererSetGizmoGeometry( renderer, gizmoVertices.data, gizmoVertices.count );
        ListFree( gizmoVertices );

        started = true;
        return true;
    }

    VulkanView::Pane VulkanView::PaneAt( QPoint position ) const {
        return position.x() < (i32)( width() * kSplitFraction ) ? Pane_Perspective : Pane_Top;
    }

    void VulkanView::Render() {
        if( !isExposed() || !EnsureStarted() ) {
            return;
        }

        // First frame has no previous timestamp to subtract, so it steps zero.
        const f32 dt = frameTimer.isValid() ? (f32)( frameTimer.restart() / 1000.0 ) : 0.0f;
        if( !frameTimer.isValid() ) {
            frameTimer.start();
        }

        const qreal dpr = devicePixelRatio();
        const i32 surfaceWidth = (i32)( width() * dpr );
        const i32 surfaceHeight = (i32)( height() * dpr );
        const i32 leftWidth = (i32)( surfaceWidth * kSplitFraction );
        const i32 rightWidth = surfaceWidth - leftWidth;

        FlyCameraUpdate( &camera, input, dt );
        // Logical height, not the pixel one: the pan deltas come from Qt cursor
        // positions, which are logical too, and mixing the two would scale
        // panning by the display's device pixel ratio.
        OrthoCameraUpdate( &topCamera, topInput, height() );

        // Deltas are per-frame: whatever the mouse did before this update has
        // been applied, so the next frame starts from zero.
        input.lookDeltaX = 0.0f;
        input.lookDeltaY = 0.0f;
        topInput.panDeltaX = 0.0f;
        topInput.panDeltaY = 0.0f;
        topInput.zoomTicks = 0.0f;

        // Each pane's projection uses its own pixel size, or the halves would
        // both be stretched by the full surface's aspect.
        RenderView views[2] = {};
        views[0].x = 0.0f;
        views[0].y = 0.0f;
        views[0].width = kSplitFraction;
        views[0].height = 1.0f;
        views[0].viewProjection = FlyCameraViewProjection( camera, leftWidth, surfaceHeight );

        views[1].x = kSplitFraction;
        views[1].y = 0.0f;
        views[1].width = 1.0f - kSplitFraction;
        views[1].height = 1.0f;
        views[1].viewProjection = OrthoCameraViewProjection( topCamera, rightWidth, surfaceHeight );

        // However many times the vertex moved since the last frame, the mesh
        // and its cage are rebuilt once.
        if( editGeometryDirty ) {
            editGeometryDirty = false;
            WorldRebuildPrimitive( world, renderer, editPrimitive );
            RefreshEditOverlay();
        }

        UpdateGizmo();

        RendererSetViews( renderer, views, 2 );
        RendererDrawFrame( renderer );

        // Presenting is FIFO, so this self-scheduling loop paces itself on vsync
        // instead of spinning the Qt event loop.
        requestUpdate();
    }

    void VulkanView::SetMovementKey( int key, bool pressed ) {
        switch( key ) {
            case Qt::Key_W:         input.forward = pressed; break;
            case Qt::Key_S:         input.back = pressed;    break;
            case Qt::Key_D:         input.right = pressed;   break;
            case Qt::Key_A:         input.left = pressed;    break;
            case Qt::Key_Space:     input.up = pressed;      break;
            case Qt::Key_Control:   input.down = pressed;    break;
            case Qt::Key_Shift:     input.fast = pressed;    break;
            default: break;
        }
    }

    void VulkanView::BeginDrag( Pane pane ) {
        if( dragging ) {
            return;
        }
        dragging = true;
        dragPane = pane;

        // Anchor where the drag started and hide the pointer, so the cursor
        // does not wander into the other pane or hit a screen edge mid-drag.
        dragAnchor = QCursor::pos();
        setCursor( Qt::BlankCursor );

        input.looking = pane == Pane_Perspective;
        topInput.panning = pane == Pane_Top;
    }

    void VulkanView::EndDrag() {
        if( !dragging ) {
            return;
        }
        dragging = false;
        input.looking = false;
        topInput.panning = false;

        unsetCursor();
        QCursor::setPos( dragAnchor );
    }

    Vec3 VulkanView::OrthoWorldAt( QPoint position ) const {
        const i32 splitX = (i32)( width() * kSplitFraction );
        const i32 paneWidth = width() - splitX;
        return OrthoCameraScreenToWorld( topCamera, (f32)( position.x() - splitX ),
                                         (f32)position.y(), paneWidth, height() );
    }

    void VulkanView::RayAt( QPoint position, Pane pane, Vec3 * outOrigin, Vec3 * outDirection ) const {
        const i32 splitX = (i32)( width() * kSplitFraction );

        if( pane == Pane_Perspective ) {
            FlyCameraScreenRay( camera, (f32)position.x(), (f32)position.y(),
                                splitX, height(), outOrigin, outDirection );
        } else {
            OrthoCameraScreenRay( topCamera, (f32)( position.x() - splitX ), (f32)position.y(),
                                  width() - splitX, height(), outOrigin, outDirection );
        }
    }

    bool VulkanView::PickAt( QPoint position, Pane pane, i32 * outPrimitive ) const {
        Vec3 origin = {};
        Vec3 direction = {};
        RayAt( position, pane, &origin, &direction );
        return WorldPick( world, renderer, origin, direction, outPrimitive );
    }

    Mat4 VulkanView::PaneViewProjection( Pane pane ) const {
        const i32 splitX = (i32)( width() * kSplitFraction );
        if( pane == Pane_Perspective ) {
            return FlyCameraViewProjection( camera, splitX, height() );
        }
        return OrthoCameraViewProjection( topCamera, width() - splitX, height() );
    }

    bool VulkanView::PickVertexAt( QPoint position, Pane pane, i32 * outVertex ) const {
        if( editPrimitive == kNoPrimitive ) {
            return false;
        }

        const i32 splitX = (i32)( width() * kSplitFraction );
        const f32 paneX = pane == Pane_Perspective ? 0.0f : (f32)splitX;
        const f32 paneWidth = pane == Pane_Perspective ? (f32)splitX : (f32)( width() - splitX );
        const f32 paneHeight = (f32)height();
        const Mat4 viewProjection = PaneViewProjection( pane );

        const i32 vertexCount = world.primitives[editPrimitive].halfMesh.vertices.count;
        i32 best = kHMNone;
        f32 bestDistance = kVertexPickRadius;
        f32 bestDepth = 0.0f;

        for( i32 v = 0; v < vertexCount; v++ ) {
            Vec3 worldPosition = {};
            if( !WorldGetVertexPosition( world, editPrimitive, v, &worldPosition ) ) {
                continue;
            }

            const Vec4 clip = viewProjection * Vec4{ worldPosition.x, worldPosition.y, worldPosition.z, 1.0f };
            // Behind the camera, where the divide would mirror it back on screen.
            if( clip.w <= 0.0f ) {
                continue;
            }

            // The renderer's negative viewport height puts clip +y at the top,
            // so screen y runs the other way from it.
            const f32 screenX = paneX + ( clip.x / clip.w * 0.5f + 0.5f ) * paneWidth;
            const f32 screenY = ( 0.5f - clip.y / clip.w * 0.5f ) * paneHeight;
            const f32 dx = screenX - (f32)position.x();
            const f32 dy = screenY - (f32)position.y();
            const f32 distance = sqrtf( dx * dx + dy * dy );
            const f32 depth = clip.z / clip.w;

            // Vertices stacked on screen - every corner of a cube seen from the
            // top - are a tie on distance, and the one nearest the camera wins.
            constexpr f32 kTie = 0.5f;
            const bool closer = distance < bestDistance - kTie;
            const bool tiedAndNearer = distance < bestDistance + kTie && best != kHMNone && depth < bestDepth;
            if( distance <= kVertexPickRadius && ( best == kHMNone || closer || tiedAndNearer ) ) {
                best = v;
                bestDistance = distance;
                bestDepth = depth;
            }
        }

        *outVertex = best;
        return best != kHMNone;
    }

    void VulkanView::SetGizmoMode( GizmoMode mode ) {
        // The same key twice is how you put the gizmo away.
        gizmo.mode = gizmo.mode == mode ? GizmoMode_None : mode;
        if( gizmo.mode == GizmoMode_None ) {
            GizmoEndDrag( gizmo );
            gizmo.hovered = GizmoAxis_None;
        }
    }

    bool VulkanView::GizmoSubject( Transform * outTransform ) const {
        if( gizmo.mode == GizmoMode_None ) {
            return false;
        }

        if( editPrimitive == kNoPrimitive ) {
            return WorldGetPrimitiveTransform( world, world.selected, outTransform );
        }

        // In edit mode the object itself stays put; only a selected vertex
        // can be moved, and a vertex has no orientation to rotate.
        if( gizmo.mode != GizmoMode_Translate || editVertex == kHMNone ) {
            return false;
        }

        Transform transform = TransformDefault();
        if( !WorldGetVertexPosition( world, editPrimitive, editVertex, &transform.position ) ) {
            return false;
        }
        *outTransform = transform;
        return true;
    }

    void VulkanView::UpdateGizmo() {
        Transform transform = {};
        const bool show = GizmoSubject( &transform );

        RendererSetGizmoVisible( renderer, show );
        if( !show ) {
            GizmoEndDrag( gizmo );
            gizmo.hovered = GizmoAxis_None;
            return;
        }

        // Not updated mid-drag: the handle has to stay where it was grabbed, or
        // moving the object would move the axis the drag is measured against.
        if( gizmo.active == GizmoAxis_None ) {
            gizmo.center = transform.position;
            // Sized off the perspective camera, which is where objects are
            // mostly handled. The top-down pane draws it at that same world
            // size rather than one of its own.
            gizmo.scale = GizmoScaleFor( gizmo.center, camera.position );
        }

        RenderGizmoRange ranges[kGizmoRangeCount] = {};
        const i32 rangeCount = GizmoDrawRanges( gizmo, ranges );
        RendererSetGizmoDraw( renderer, GizmoDrawTransform( gizmo ), ranges, rangeCount );
    }

    bool VulkanView::BeginGizmoDrag( QPoint position, Pane pane ) {
        Transform transform = {};
        if( !GizmoSubject( &transform ) ) {
            return false;
        }

        Vec3 origin = {};
        Vec3 direction = {};
        RayAt( position, pane, &origin, &direction );

        const GizmoAxis axis = GizmoPick( gizmo, origin, direction );
        if( axis == GizmoAxis_None ) {
            return false;
        }

        return GizmoBeginDrag( gizmo, axis, transform, origin, direction );
    }

    void VulkanView::ArmCreate( QPoint position ) {
        if( !started ) {
            return;
        }

        // Nothing is built yet. Whether this press is a click that clears the
        // selection or the start of a new plane is not knowable until the
        // mouse either moves far enough or comes back up.
        createPending = true;
        createPressPosition = position;
    }

    void VulkanView::BeginCreate( QPoint position ) {
        if( createPrimitive != kNoPrimitive || !started ) {
            return;
        }

        createPending = false;

        const f32 step = renderer->gridSpacing;
        createStart = Vec3SnapTo( OrthoWorldAt( position ), step );

        // A unit quad centred on its own origin, so the transform alone can
        // place and size it. The mesh never has to be rebuilt while dragging.
        HalfMesh quad = {};
        HalfMeshCreateQuad( quad, 1.0f );

        RenderMaterial material = RenderMaterialDefault();
        material.albedo = Vec3{ 0.45f, 0.62f, 0.50f };

        createPrimitive = WorldAddPrimitive( world, renderer, quad, material, TransformDefault() );
        HalfMeshFree( quad );

        if( createPrimitive == kNoPrimitive ) {
            fprintf( stderr, "Failed to create a plane\n" );
            return;
        }

        // Whatever you just made is what you are working on.
        WorldSetSelected( world, renderer, createPrimitive );
        UpdateCreate( position );
    }

    void VulkanView::UpdateCreate( QPoint position ) {
        if( createPrimitive == kNoPrimitive ) {
            return;
        }

        const f32 step = renderer->gridSpacing;
        const Vec3 corner = Vec3SnapTo( OrthoWorldAt( position ), step );

        f32 minX = Min( createStart.x, corner.x );
        f32 maxX = Max( createStart.x, corner.x );
        f32 minZ = Min( createStart.z, corner.z );
        f32 maxZ = Max( createStart.z, corner.z );

        // A press that never moves still has to produce something visible, so
        // the smallest plane is one cell rather than nothing.
        if( maxX - minX < step ) { maxX = minX + step; }
        if( maxZ - minZ < step ) { maxZ = minZ + step; }

        Transform transform = TransformDefault();
        transform.position = Vec3{ 0.5f * ( minX + maxX ), 0.0f, 0.5f * ( minZ + maxZ ) };
        transform.scale = Vec3{ maxX - minX, 1.0f, maxZ - minZ };

        WorldSetPrimitiveTransform( world, renderer, createPrimitive, transform );
    }

    void VulkanView::EndCreate() {
        // Disarms a press that never travelled, which is what leaves a plain
        // click having done nothing but clear the selection.
        createPending = false;
        createPrimitive = kNoPrimitive;
    }

    void VulkanView::DeleteSelected() {
        if( !started || world.selected == kNoPrimitive ) {
            return;
        }

        const i32 removed = world.selected;
        if( !WorldRemovePrimitive( world, renderer, removed ) ) {
            return;
        }

        // A plane being dragged out right now is a primitive like any other, so
        // the index tracking it has to follow the removal - and stop naming
        // anything at all if it was what just went.
        createPrimitive = WorldRemapPrimitive( createPrimitive, removed );

        // Same for the cage, which additionally has to come down if what it was
        // describing is what was deleted.
        editPrimitive = WorldRemapPrimitive( editPrimitive, removed );
        RefreshEditOverlay();
    }

    void VulkanView::ToggleEditMode() {
        if( !started ) {
            return;
        }

        if( editPrimitive != kNoPrimitive ) {
            // The highlight goes back on the way out, or the object would stop
            // reading as selected the moment its cage came down.
            const i32 previous = editPrimitive;
            editPrimitive = kNoPrimitive;
            editVertex = kHMNone;
            WorldSetPrimitiveHighlight( world, renderer, previous, true );
        } else {
            // Nothing selected is nothing to edit, so Tab is a no-op rather
            // than a mode with no subject.
            if( world.selected == kNoPrimitive ) {
                return;
            }
            editPrimitive = world.selected;
            editVertex = kHMNone;
            // The cage shows which object is the subject far better than a
            // tint does, so the tint comes off rather than competing with it.
            WorldSetPrimitiveHighlight( world, renderer, editPrimitive, false );
        }

        RefreshEditOverlay();
    }

    void VulkanView::RefreshEditOverlay() {
        // A primitive that can no longer produce a cage takes the mode down
        // with it, so edit mode never outlives what it was editing.
        if( editPrimitive == kNoPrimitive || !WorldSetEditOverlay( world, renderer, editPrimitive, editVertex ) ) {
            // A cage that could not be built leaves edit mode off, so whatever
            // was about to wear it gets its highlight back rather than sitting
            // selected with nothing showing it.
            const i32 previous = editPrimitive;
            editPrimitive = kNoPrimitive;
            editVertex = kHMNone;
            editGeometryDirty = false;
            RendererClearEditOverlay( renderer );
            WorldSetPrimitiveHighlight( world, renderer, previous, true );
        }
    }

    void VulkanView::keyPressEvent( QKeyEvent * event ) {
        // Auto-repeat would otherwise deliver a release/press pair per repeat,
        // which reads as the key stuttering rather than being held.
        if( !event->isAutoRepeat() ) {
            SetMovementKey( event->key(), true );

            const i32 step = event->key() - Qt::Key_1;
            if( step >= 0 && step < kGridStepCount && started ) {
                RendererSetGridSpacing( renderer, kGridSteps[step] );
            }

            if( event->key() == Qt::Key_Delete ) {
                DeleteSelected();
            }

            if( event->key() == Qt::Key_Tab ) {
                ToggleEditMode();
            }

            if( event->key() == Qt::Key_T ) {
                SetGizmoMode( GizmoMode_Translate );
            }

            if( event->key() == Qt::Key_R ) {
                SetGizmoMode( GizmoMode_Rotate );
            }
        }
        QWindow::keyPressEvent( event );
    }

    void VulkanView::keyReleaseEvent( QKeyEvent * event ) {
        if( !event->isAutoRepeat() ) {
            SetMovementKey( event->key(), false );
        }
        QWindow::keyReleaseEvent( event );
    }

    void VulkanView::mousePressEvent( QMouseEvent * event ) {
        // Keys only reach a QWindow that holds focus, and clicking the
        // viewport is how the user expects to hand it over.
        requestActivate();

        const QPoint position = event->position().toPoint();
        if( event->button() == Qt::RightButton ) {
            BeginDrag( PaneAt( position ) );
        } else if( event->button() == Qt::LeftButton ) {
            // A press on a handle is a drag of the selection, never a pick of
            // whatever happens to lie behind it.
            if( BeginGizmoDrag( position, PaneAt( position ) ) ) {
                QWindow::mousePressEvent( event );
                return;
            }

            // Edit mode locks on to its subject. While the cage is up a left
            // click is not a way to pick a different object, nor to place a new
            // one - both would move the selection out from under the cage.
            // It picks a vertex of the subject instead, and clicking away from
            // every vertex drops the one selected. Tab puts the cage away and
            // hands object picking back.
            if( editPrimitive != kNoPrimitive ) {
                i32 vertex = kHMNone;
                PickVertexAt( position, PaneAt( position ), &vertex );
                if( vertex != editVertex ) {
                    editVertex = vertex;
                    RefreshEditOverlay();
                }
            } else {
                // Selecting wins over creating: a click that lands on something
                // picks it, and only empty space starts a new plane. Placing one
                // on top of another therefore needs the space cleared first.
                const Pane pane = PaneAt( position );
                i32 hit = kNoPrimitive;
                if( PickAt( position, pane, &hit ) ) {
                    WorldSetSelected( world, renderer, hit );
                } else {
                    // Empty space always clears the selection. It only becomes a
                    // new plane if the press turns into a drag, so a click on
                    // nothing deselects and leaves the scene alone.
                    WorldSetSelected( world, renderer, kNoPrimitive );
                    if( pane == Pane_Top ) {
                        ArmCreate( position );
                    }
                }
            }
        }
        QWindow::mousePressEvent( event );
    }

    void VulkanView::mouseReleaseEvent( QMouseEvent * event ) {
        if( event->button() == Qt::RightButton ) {
            EndDrag();
        } else if( event->button() == Qt::LeftButton ) {
            EndCreate();
            GizmoEndDrag( gizmo );
        }
        QWindow::mouseReleaseEvent( event );
    }

    void VulkanView::mouseMoveEvent( QMouseEvent * event ) {
        const QPoint position = event->position().toPoint();

        if( gizmo.mode != GizmoMode_None && !dragging ) {
            Vec3 origin = {};
            Vec3 direction = {};
            RayAt( position, PaneAt( position ), &origin, &direction );

            if( gizmo.active != GizmoAxis_None ) {
                Transform transform = {};
                if( GizmoUpdateDrag( gizmo, origin, direction, renderer->gridSpacing, &transform ) ) {
                    if( editPrimitive != kNoPrimitive ) {
                        if( WorldSetVertexPosition( world, editPrimitive, editVertex, transform.position ) ) {
                            editGeometryDirty = true;
                        }
                    } else {
                        WorldSetPrimitiveTransform( world, renderer, world.selected, transform );
                    }
                }
            } else {
                gizmo.hovered = GizmoPick( gizmo, origin, direction );
            }
        }

        // The platform's own click-versus-drag threshold, so this matches what
        // every other application on the machine considers a drag. Measured
        // from the press, and skipped while a camera drag is warping the
        // cursor around, which would otherwise read as enormous travel.
        if( createPending && !dragging ) {
            const i32 travel = ( position - createPressPosition ).manhattanLength();
            if( travel >= QGuiApplication::styleHints()->startDragDistance() ) {
                // Anchored at the press, not here, or the plane would start
                // from wherever the cursor happened to cross the threshold.
                BeginCreate( createPressPosition );
            }
        }

        // Creating reads the real cursor position, so unlike the camera drags
        // it must not warp the pointer back to an anchor.
        if( createPrimitive != kNoPrimitive ) {
            UpdateCreate( position );
        }

        if( dragging ) {
            const QPoint global = event->globalPosition().toPoint();
            const QPoint delta = global - dragAnchor;
            // The warp below generates its own move event landing exactly on
            // the anchor; ignoring a zero delta is what stops it recursing.
            if( !delta.isNull() ) {
                if( dragPane == Pane_Perspective ) {
                    input.lookDeltaX += (f32)delta.x();
                    input.lookDeltaY += (f32)delta.y();
                } else {
                    topInput.panDeltaX += (f32)delta.x();
                    topInput.panDeltaY += (f32)delta.y();
                }
                QCursor::setPos( dragAnchor );
            }
        }
        QWindow::mouseMoveEvent( event );
    }

    void VulkanView::wheelEvent( QWheelEvent * event ) {
        // Zoom belongs to the pane under the cursor, and only the orthographic
        // one has a zoom to speak of.
        if( PaneAt( event->position().toPoint() ) == Pane_Top ) {
            // A notch is 120 eighths of a degree by Qt's convention.
            topInput.zoomTicks += (f32)event->angleDelta().y() / 120.0f;
        }
        QWindow::wheelEvent( event );
    }

    void VulkanView::focusOutEvent( QFocusEvent * event ) {
        // Releases arrive at whoever has focus, so a key or button still down
        // when focus leaves would otherwise stick on forever.
        EndDrag();
        EndCreate();
        GizmoEndDrag( gizmo );
        input = {};
        topInput = {};
        QWindow::focusOutEvent( event );
    }

    void VulkanView::exposeEvent( QExposeEvent * event ) {
        QWindow::exposeEvent( event );
        if( isExposed() ) {
            Render();
        }
    }

    void VulkanView::resizeEvent( QResizeEvent * event ) {
        QWindow::resizeEvent( event );
        if( !started ) {
            return;
        }

        const qreal dpr = devicePixelRatio();
        RendererSetSize( renderer, (i32)( event->size().width() * dpr ), (i32)( event->size().height() * dpr ) );
    }

    bool VulkanView::event( QEvent * event ) {
        if( event->type() == QEvent::UpdateRequest ) {
            Render();
            return true;
        }
        return QWindow::event( event );
    }

} // namespace sol
