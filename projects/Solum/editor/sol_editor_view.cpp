#include "sol_editor_view.h"

#include <QCursor>
#include <QFocusEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QVulkanInstance>

#include <cstdio>

namespace sol {

    VulkanView::VulkanView( Renderer * renderer )
        : renderer( renderer ), started( false ), startFailed( false ),
          camera( FlyCameraDefault() ), input( {} ), lookAnchor(), frameTimer() {
        setSurfaceType( QSurface::VulkanSurface );
    }

    VulkanView::~VulkanView() {
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

        RendererAddDebugTriangle( renderer );

        started = true;
        return true;
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

        FlyCameraUpdate( &camera, input, dt );
        // Deltas are per-frame: whatever the mouse did before this update has
        // been applied, so the next frame starts from zero.
        input.lookDeltaX = 0.0f;
        input.lookDeltaY = 0.0f;

        const qreal dpr = devicePixelRatio();
        RendererSetViewProjection( renderer,
                                   FlyCameraViewProjection( camera, (i32)( width() * dpr ),
                                                            (i32)( height() * dpr ) ) );

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

    void VulkanView::SetLooking( bool looking ) {
        if( looking == input.looking ) {
            return;
        }
        input.looking = looking;

        if( looking ) {
            // Anchor where the drag started and hide the pointer, so the cursor
            // does not wander off the viewport or hit a screen edge mid-turn.
            lookAnchor = QCursor::pos();
            setCursor( Qt::BlankCursor );
        } else {
            unsetCursor();
            QCursor::setPos( lookAnchor );
        }
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
            SetLooking( true );
        }
        QWindow::mousePressEvent( event );
    }

    void VulkanView::mouseReleaseEvent( QMouseEvent * event ) {
        if( event->button() == Qt::RightButton ) {
            SetLooking( false );
        }
        QWindow::mouseReleaseEvent( event );
    }

    void VulkanView::mouseMoveEvent( QMouseEvent * event ) {
        if( input.looking ) {
            const QPoint global = event->globalPosition().toPoint();
            const QPoint delta = global - lookAnchor;
            // The warp below generates its own move event landing exactly on
            // the anchor; ignoring a zero delta is what stops it recursing.
            if( !delta.isNull() ) {
                input.lookDeltaX += (f32)delta.x();
                input.lookDeltaY += (f32)delta.y();
                QCursor::setPos( lookAnchor );
            }
        }
        QWindow::mouseMoveEvent( event );
    }

    void VulkanView::focusOutEvent( QFocusEvent * event ) {
        // Releases arrive at whoever has focus, so a key or button still down
        // when focus leaves would otherwise stick on forever.
        const bool wasLooking = input.looking;
        input = {};
        if( wasLooking ) {
            unsetCursor();
            QCursor::setPos( lookAnchor );
        }
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
