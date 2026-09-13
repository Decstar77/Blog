#include "sol_editor_view.h"

#include <QCursor>
#include <QFocusEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QVulkanInstance>
#include <QWheelEvent>

#include <cstdio>

namespace sol {

    // Fraction of the surface the perspective pane gets; the top-down pane
    // takes the rest.
    constexpr f32 kSplitFraction = 0.5f;

    VulkanView::VulkanView( Renderer * renderer )
        : renderer( renderer ), world(), started( false ), startFailed( false ),
          camera( FlyCameraDefault() ), topCamera( OrthoCameraDefault( OrthoAxis_Top ) ),
          input(), topInput(), dragging( false ), dragPane( Pane_Perspective ),
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
        OrthoCameraUpdate( &topCamera, topInput, surfaceHeight );

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

    void VulkanView::keyPressEvent( QKeyEvent * event ) {
        // Auto-repeat would otherwise deliver a release/press pair per repeat,
        // which reads as the key stuttering rather than being held.
        if( !event->isAutoRepeat() ) {
            SetMovementKey( event->key(), true );
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
        if( event->button() == Qt::RightButton ) {
            // Keys only reach a QWindow that holds focus, and clicking the
            // viewport is how the user expects to hand it over.
            requestActivate();
            BeginDrag( PaneAt( event->position().toPoint() ) );
        }
        QWindow::mousePressEvent( event );
    }

    void VulkanView::mouseReleaseEvent( QMouseEvent * event ) {
        if( event->button() == Qt::RightButton ) {
            EndDrag();
        }
        QWindow::mouseReleaseEvent( event );
    }

    void VulkanView::mouseMoveEvent( QMouseEvent * event ) {
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
