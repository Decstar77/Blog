#pragma once

#include "sol_camera.h"
#include "sol_render.h"

#include <QElapsedTimer>
#include <QPoint>
#include <QWindow>

namespace sol {

    // A QWindow with a Vulkan surface type, so Qt creates the native window and
    // the VkSurfaceKHR and the engine renderer draws straight into it. Wrapped
    // by QWidget::createWindowContainer to sit inside ordinary Qt layouts.
    class VulkanView : public QWindow {
    public:
        explicit VulkanView( Renderer * renderer );
        ~VulkanView() override;

        bool startupFailed() const { return startFailed; }

    protected:
        void exposeEvent( QExposeEvent * event ) override;
        void resizeEvent( QResizeEvent * event ) override;
        void keyPressEvent( QKeyEvent * event ) override;
        void keyReleaseEvent( QKeyEvent * event ) override;
        void mousePressEvent( QMouseEvent * event ) override;
        void mouseReleaseEvent( QMouseEvent * event ) override;
        void mouseMoveEvent( QMouseEvent * event ) override;
        void focusOutEvent( QFocusEvent * event ) override;
        bool event( QEvent * event ) override;

    private:
        bool EnsureStarted();
        void Render();

        // Qt delivers movement keys as press/release edges rather than the
        // polled state GLFW gives, so held keys are tracked here and handed to
        // the shared camera as one input snapshot per frame.
        void SetMovementKey( int key, bool pressed );
        void SetLooking( bool looking );

        Renderer *      renderer;
        bool            started;
        bool            startFailed;

        FlyCamera       camera;
        // Only the held-key and look flags persist between frames; the mouse
        // deltas are accumulated from events and consumed by each Render.
        FlyCameraInput  input;
        // Screen position the cursor is warped back to while looking, which is
        // how an unbounded drag is emulated without GLFW's disabled-cursor mode.
        QPoint          lookAnchor;
        QElapsedTimer   frameTimer;
    };

} // namespace sol
