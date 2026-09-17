#pragma once

#include "sol_camera.h"
#include "sol_editor_gizmo.h"
#include "sol_editor_grid.h"
#include "sol_render.h"
#include "sol_world.h"

#include <QElapsedTimer>
#include <QPoint>
#include <QRect>
#include <QWindow>

namespace sol {

    // Which camera a pane drives. The two kinds answer the same questions -
    // a ray through a pixel, a view projection - so everything downstream of
    // a pane only has to branch here.
    enum PaneKind {
        PaneKind_Perspective,
        PaneKind_Ortho,
    };

    // One viewport rectangle and the camera behind it. Panes hold their own
    // input accumulators because a drag belongs to the pane it started in,
    // not to the window.
    struct EditorPane {
        PaneKind            kind;
        // Fraction of the widget, origin top left, matching RenderView. A
        // layout change rewrites these and nothing else, so the cameras keep
        // wherever the user had put them.
        f32                 x;
        f32                 y;
        f32                 width;
        f32                 height;

        FlyCamera           fly;
        FlyCameraInput      flyInput;
        OrthoCamera         ortho;
        OrthoCameraInput    orthoInput;
    };

    // The pane slots are fixed in role - 0 perspective, 1 top, 2 front,
    // 3 side - and a layout only decides how many of them are live and where
    // they sit. That is what lets switching layouts preserve every camera.
    enum PaneLayout {
        PaneLayout_Single,
        PaneLayout_Split,
        PaneLayout_Quad,
        PaneLayout_Count,
    };

    // Build mode is two operations back to back, and the stage is what says
    // which one a mouse event belongs to. Off and Ready are the same mode from
    // the scene's point of view - nothing is half-built - but only Ready takes
    // a press as the start of a box.
    enum BuildStage {
        BuildStage_Off,     // not in build mode; clicks select as usual
        BuildStage_Ready,   // in build mode, waiting for the press that starts a base
        BuildStage_Base,    // dragging the base rectangle out, button still down
        BuildStage_Height,  // base locked, moving the mouse extrudes, a click commits
    };

    class VulkanView : public QWindow {
    public:
        explicit VulkanView( Renderer * renderer );
        ~VulkanView() override;

        bool startupFailed() const { return startFailed; }

        // Public so the main window's View menu can drive it alongside the
        // function keys.
        void SetLayout( PaneLayout next );
        PaneLayout CurrentLayout() const { return layout; }

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
        bool EnsureStarted();
        void Render();

        // Pane index of whatever is under the point. Never fails: a point
        // outside every rect - which rounding at a seam can produce - lands on
        // the active pane rather than on nothing.
        i32 PaneAt( QPoint position ) const;
        // Logical pixels, which is the space Qt hands mouse positions in.
        QRect PaneRect( i32 pane ) const;
        void SetActivePane( i32 pane );
        // The pane whose camera WASD drives: the active one when it is a
        // perspective pane, otherwise the first that is.
        i32 MovementPane() const;
        void SetMovementKey( int key, bool pressed );
        void BeginDrag( i32 pane );
        void EndDrag();

        void RayAt( QPoint position, i32 pane, Vec3 * outOrigin, Vec3 * outDirection ) const;
        bool PickAt( QPoint position, i32 pane, i32 * outPrimitive ) const;

        Mat4 PaneViewProjection( i32 pane ) const;
        bool PickVertexAt( QPoint position, i32 pane, i32 * outVertex ) const;
        void SetGizmoMode( GizmoMode mode );
        bool GizmoSubject( Transform * outTransform ) const;
        void UpdateGizmo();
        bool BeginGizmoDrag( QPoint position, i32 pane );

        // Moves the grid onto the next of the three axis planes. Everything
        // that reads the grid - snapping, placing, the drawn lines - follows
        // from the one field this writes.
        void CycleGridPlane();
        // Grid coordinates of whatever a pixel is pointing at on the grid.
        // False when that pane cannot see the grid plane at all.
        bool GridPointAt( QPoint position, i32 pane, Vec3 * outLocal ) const;

        void ToggleBuildMode();
        // Presses and moves, routed by the stage. Each returns whether build
        // mode consumed the event, so the ordinary select-and-place path can
        // be skipped without testing the stage twice.
        bool BuildMousePress( QPoint position, i32 pane );
        void BuildMouseMove( QPoint position );
        void BuildMouseRelease();
        // Writes the box's transform from the base rectangle and the height.
        // CPU-only, so it is safe to call on every mouse move.
        void ApplyBuildTransform();
        // Swaps the flat base quad for a cube once the base is locked. Rebuilds
        // the GPU mesh, so it happens once, on the release that locks it.
        void BuildToBox();
        // Keeps whatever has been built and returns to Ready, so one B gets you
        // as many boxes as you want.
        void CommitBuild();
        // Throws the half-built primitive away. Leaves build mode alone: a
        // cancel mid-box drops back to Ready, not out of the mode.
        void CancelBuild();

        void DeleteSelected();

        void ToggleEditMode();
        void RefreshEditOverlay();

        Renderer *          renderer;

        World               world;
        bool                started;
        bool                startFailed;

        EditorPane          panes[kMaxRenderViews];
        i32                 paneCount;
        PaneLayout          layout;
        // Where the keyboard goes, and the fallback for a point that misses
        // every rect.
        i32                 activePane;
        // Held on the window rather than on a pane: the keys are down for as
        // long as they are down, whichever pane is spending them.
        FlyCameraInput      movement;

        bool                dragging;
        i32                 dragPane;

        // The plane every placement snaps to. The renderer still draws its grid
        // on y = 0, so this starts matching it; everything that places geometry
        // goes through here rather than through world axes, which is what will
        // let the grid be re-aimed later.
        EditorGrid          grid;

        BuildStage          buildStage;
        i32                 buildPrimitive;
        i32                 buildPane;
        // Base rectangle in grid coordinates, min and max on each in-plane
        // axis. z is unused - the base is on the plane by construction.
        Vec3                buildStart;
        Vec3                buildMin;
        Vec3                buildMax;
        // Signed, along the grid normal, so a box can be pulled down as well as
        // up. Magnitude is never below one cell.
        f32                 buildHeight;

        i32                 editPrimitive;
        i32                 editVertex;
        bool                editGeometryDirty;

        Gizmo               gizmo;
        QPoint              dragAnchor;
        QElapsedTimer       frameTimer;
    };

} // namespace sol
