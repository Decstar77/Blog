#pragma once

#include "sol_camera.h"
#include "sol_render.h"
#include "sol_world.h"

#include <QElapsedTimer>
#include <QPoint>
#include <QWindow>

namespace sol {

    // A QWindow with a Vulkan surface type, so Qt creates the native window and
    // the VkSurfaceKHR and the engine renderer draws straight into it. Wrapped
    // by QWidget::createWindowContainer to sit inside ordinary Qt layouts.
    //
    // The surface is split into two panes drawn from the same scene: a
    // perspective fly camera on the left and a top-down orthographic camera on
    // the right. Both are one swapchain and one render pass - see RenderView.
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
        void wheelEvent( QWheelEvent * event ) override;
        void focusOutEvent( QFocusEvent * event ) override;
        bool event( QEvent * event ) override;

    private:
        enum Pane {
            Pane_Perspective,
            Pane_Top,
        };

        bool EnsureStarted();
        void Render();

        // Which pane a point in window coordinates falls in. The split is down
        // the middle, matching the view rectangles handed to the renderer.
        Pane PaneAt( QPoint position ) const;

        // Qt delivers movement keys as press/release edges rather than the
        // polled state GLFW gives, so held keys are tracked here and handed to
        // the shared camera as one input snapshot per frame.
        void SetMovementKey( int key, bool pressed );
        // A drag belongs to whichever pane it started in, so leaving that pane
        // mid-drag does not hand the camera over to its neighbour.
        void BeginDrag( Pane pane );
        void EndDrag();

        Renderer *          renderer;
        // The authored scene. Owns the half-meshes; the renderer owns the
        // triangles built from them.
        World               world;
        bool                started;
        bool                startFailed;

        FlyCamera           camera;
        OrthoCamera         topCamera;
        // Only the held-key and drag flags persist between frames; the mouse
        // deltas are accumulated from events and consumed by each Render.
        FlyCameraInput      input;
        OrthoCameraInput    topInput;

        bool                dragging;
        Pane                dragPane;
        // Screen position the cursor is warped back to while dragging, which is
        // how an unbounded drag is emulated without GLFW's disabled-cursor mode.
        QPoint              dragAnchor;
        QElapsedTimer       frameTimer;
    };

} // namespace sol
