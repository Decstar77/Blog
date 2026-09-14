#pragma once

#include "sol_camera.h"
#include "sol_editor_gizmo.h"
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

        // Where a point in the window sits on the top-down camera's plane.
        // Everything here is in Qt's logical units, mouse position included.
        Vec3 OrthoWorldAt( QPoint position ) const;

        // The cursor ray through whichever pane the point falls in.
        void RayAt( QPoint position, Pane pane, Vec3 * outOrigin, Vec3 * outDirection ) const;

        // Nearest primitive under the cursor, picked through whichever pane the
        // point falls in.
        bool PickAt( QPoint position, Pane pane, i32 * outPrimitive ) const;

        // T and R put the translate and rotate gizmo up on the selection, and
        // pressing the same key again takes it down.
        void SetGizmoMode( GizmoMode mode );
        // Moves the gizmo onto the selection and sizes it for this frame. Also
        // what makes it pickable, since picking reads that centre and size.
        void UpdateGizmo();
        // True when the press was taken by a gizmo handle, in which case it is
        // not also a selection click.
        bool BeginGizmoDrag( QPoint position, Pane pane );

        // Left-dragging in the top-down pane pulls out a plane. Nothing is
        // created on press - see ArmCreate - and once it is, it starts as a
        // unit quad resized purely through its transform, so dragging costs a
        // matrix rather than a mesh rebuild.
        //
        // position is where the press landed, not where the cursor is now:
        // the plane has to span from the corner the user started at.
        void ArmCreate( QPoint position );
        void BeginCreate( QPoint position );
        void UpdateCreate( QPoint position );
        void EndCreate();

        // Removes whatever is selected, leaving nothing selected. Bound to the
        // Delete key; a no-op when the selection is empty.
        void DeleteSelected();

        // Tab puts the edit cage up on the selection and takes it down again.
        // Nothing selected means nothing to edit, so it does nothing.
        void ToggleEditMode();
        // Rebuilds the cage for editPrimitive, or clears it when there is no
        // longer anything to show. The one path that touches the overlay, so
        // the renderer and editPrimitive cannot disagree.
        void RefreshEditOverlay();

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

        // The plane being pulled out right now, and the snapped corner the drag
        // started from. kNoPrimitive when nothing is being created.
        i32                 createPrimitive;
        Vec3                createStart;
        // A left press on empty space arms a create rather than committing to
        // one: the plane appears only once the cursor has travelled far enough
        // to read as a drag, so a plain click just clears the selection.
        bool                createPending;
        QPoint              createPressPosition;
        // The primitive whose cage is up, or kNoPrimitive when edit mode is
        // off. While this names something it is also the locked selection:
        // clicks stop picking objects, so the subject cannot change without
        // leaving edit mode first.
        i32                 editPrimitive;
        // Drives the selection's transform. Only up in object mode: in edit
        // mode the subject is the geometry, not the object.
        Gizmo               gizmo;
        // Screen position the cursor is warped back to while dragging, which is
        // how an unbounded drag is emulated without GLFW's disabled-cursor mode.
        QPoint              dragAnchor;
        QElapsedTimer       frameTimer;
    };

} // namespace sol
