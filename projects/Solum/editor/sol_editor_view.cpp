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

    VulkanView::VulkanView( Renderer * renderer )
        : renderer( renderer ), world( WorldCreate() ), started( false ), startFailed( false ),
          camera( FlyCameraDefault() ), topCamera( OrthoCameraDefault( OrthoAxis_Top ) ),
          input(), topInput(), dragging( false ), dragPane( Pane_Perspective ),
          createPrimitive( kNoPrimitive ), createStart(),
          createPending( false ), createPressPosition(),
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

    bool VulkanView::PickAt( QPoint position, Pane pane, i32 * outPrimitive ) const {
        const i32 splitX = (i32)( width() * kSplitFraction );

        Vec3 origin = {};
        Vec3 direction = {};
        if( pane == Pane_Perspective ) {
            FlyCameraScreenRay( camera, (f32)position.x(), (f32)position.y(),
                                splitX, height(), &origin, &direction );
        } else {
            OrthoCameraScreenRay( topCamera, (f32)( position.x() - splitX ), (f32)position.y(),
                                  width() - splitX, height(), &origin, &direction );
        }

        return WorldPick( world, renderer, origin, direction, outPrimitive );
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

        createPrimitive = WorldAddPrimitive( world, renderer, quad, material, Mat4Identity() );
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

        const Vec3 center = { 0.5f * ( minX + maxX ), 0.0f, 0.5f * ( minZ + maxZ ) };
        const Vec3 scale = { maxX - minX, 1.0f, maxZ - minZ };

        WorldSetPrimitiveTransform( world, renderer, createPrimitive,
                                    Mat4Translate( center ) * Mat4Scale( scale ) );
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
        QWindow::mousePressEvent( event );
    }

    void VulkanView::mouseReleaseEvent( QMouseEvent * event ) {
        if( event->button() == Qt::RightButton ) {
            EndDrag();
        } else if( event->button() == Qt::LeftButton ) {
            EndCreate();
        }
        QWindow::mouseReleaseEvent( event );
    }

    void VulkanView::mouseMoveEvent( QMouseEvent * event ) {
        const QPoint position = event->position().toPoint();

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
