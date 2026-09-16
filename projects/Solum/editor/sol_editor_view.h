#pragma once

#include "sol_camera.h"
#include "sol_editor_gizmo.h"
#include "sol_render.h"
#include "sol_world.h"

#include <QElapsedTimer>
#include <QPoint>
#include <QWindow>

namespace sol {

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

        Pane PaneAt( QPoint position ) const;

        void SetMovementKey( int key, bool pressed );
        void BeginDrag( Pane pane );
        void EndDrag();

        Vec3 OrthoWorldAt( QPoint position ) const;
        void RayAt( QPoint position, Pane pane, Vec3 * outOrigin, Vec3 * outDirection ) const;
        bool PickAt( QPoint position, Pane pane, i32 * outPrimitive ) const;

        Mat4 PaneViewProjection( Pane pane ) const;
        bool PickVertexAt( QPoint position, Pane pane, i32 * outVertex ) const;
        void SetGizmoMode( GizmoMode mode );
        bool GizmoSubject( Transform * outTransform ) const;
        void UpdateGizmo();
        bool BeginGizmoDrag( QPoint position, Pane pane );

        void ArmCreate( QPoint position );
        void BeginCreate( QPoint position );
        void UpdateCreate( QPoint position );
        void EndCreate();

        void DeleteSelected();

        void ToggleEditMode();
        void RefreshEditOverlay();

        Renderer *          renderer;
        
        World               world;
        bool                started;
        bool                startFailed;

        FlyCamera           camera;
        OrthoCamera         topCamera;
        FlyCameraInput      input;
        OrthoCameraInput    topInput;

        bool                dragging;
        Pane                dragPane;

        i32                 createPrimitive;
        Vec3                createStart;
        bool                createPending;
        QPoint              createPressPosition;
        
        i32                 editPrimitive;
        i32                 editVertex;
        bool                editGeometryDirty;

        Gizmo               gizmo;
        QPoint              dragAnchor;
        QElapsedTimer       frameTimer;
    };

} // namespace sol
