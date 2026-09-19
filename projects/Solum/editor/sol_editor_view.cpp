#include "sol_editor_view.h"

#include <QCursor>
#include <QFocusEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QVulkanInstance>
#include <QWheelEvent>

#include <cmath>
#include <cstdio>

namespace sol {

    // The quad layout is the widest this goes, and it is exactly what the
    // renderer will draw in one frame.
    static_assert( kMaxRenderViews >= 4, "the quad layout needs four render views" );

    // Selectable grid sizes, smallest first, bound to the 1-6 keys. The snap
    // step follows whichever is current, so the grid is not decoration - it is
    // the thing geometry lands on.
    constexpr f32 kGridSteps[] = { 0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f };
    constexpr i32 kGridStepCount = (i32)( sizeof( kGridSteps ) / sizeof( kGridSteps[0] ) );

    // The three axis planes the grid can be put on, cycled by G. Anything
    // further - a grid laid on the face of an object - would come from a pick
    // rather than a key, and would land on the same EditorGrid.
    constexpr Vec3 kGridNormals[] = {
        { 0.0f, 1.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f },
        { 1.0f, 0.0f, 0.0f },
    };
    constexpr i32 kGridNormalCount = (i32)( sizeof( kGridNormals ) / sizeof( kGridNormals[0] ) );

    // The two modes that take the left button over. Neither changes what the
    // scene looks like, so without a border on the viewport there is nothing
    // on screen saying a click is about to build or edit rather than select.
    constexpr Vec3 kBuildBorderColor = { 0.58f, 0.36f, 0.16f };
    constexpr Vec3 kEditBorderColor = { 0.26f, 0.68f, 0.32f };

    // How far, in logical pixels, a click may land from a vertex and still pick it.
    constexpr f32 kVertexPickRadius = 10.0f;

    // Slot 0 is the perspective pane in every layout, which is what lets the
    // gizmo reach for a perspective camera without searching for one.
    constexpr i32 kPerspectivePane = 0;

    // Builds the cameras for all four slots once. A layout change rewrites
    // rectangles only, so a camera framed up in the quad layout is still
    // pointing the same way when that layout comes back.
    static void PanesInit( EditorPane * panes ) {
        panes[0] = {};
        panes[0].kind = PaneKind_Perspective;
        panes[0].fly = FlyCameraDefault();

        panes[1] = {};
        panes[1].kind = PaneKind_Ortho;
        panes[1].ortho = OrthoCameraDefault( OrthoAxis_Top );

        panes[2] = {};
        panes[2].kind = PaneKind_Ortho;
        panes[2].ortho = OrthoCameraDefault( OrthoAxis_Front );

        panes[3] = {};
        panes[3].kind = PaneKind_Ortho;
        panes[3].ortho = OrthoCameraDefault( OrthoAxis_Side );
    }

    static void PaneSetRect( EditorPane & pane, f32 x, f32 y, f32 width, f32 height ) {
        pane.x = x;
        pane.y = y;
        pane.width = width;
        pane.height = height;
    }

    // Movement keys live on the window while look deltas live on the pane, so
    // the two halves of a fly camera's input meet here, once per frame, for
    // whichever pane the keyboard is currently spending itself on.
    static FlyCameraInput FlyInputCombine( const FlyCameraInput & pane, const FlyCameraInput & movement ) {
        FlyCameraInput input = movement;
        input.looking = pane.looking;
        input.lookDeltaX = pane.lookDeltaX;
        input.lookDeltaY = pane.lookDeltaY;
        return input;
    }

    // Axis-aligned rectangle in grid coordinates spanned by two snapped
    // corners. Never smaller than one cell: a press that never travelled still
    // has to produce something you can see.
    static void GridRect( const EditorGrid & grid, Vec3 a, Vec3 b, Vec3 * outMin, Vec3 * outMax ) {
        Vec3 min = Vec3{ Min( a.x, b.x ), Min( a.y, b.y ), 0.0f };
        Vec3 max = Vec3{ Max( a.x, b.x ), Max( a.y, b.y ), 0.0f };
        if( max.x - min.x < grid.step ) { max.x = min.x + grid.step; }
        if( max.y - min.y < grid.step ) { max.y = min.y + grid.step; }
        *outMin = min;
        *outMax = max;
    }

    // Places a unit primitive - the base quad, or the cube it becomes - across a
    // grid rectangle. Both are centred on their own origin and lie in local xz
    // with local +y up, which is exactly the space EditorGridRotation lands on
    // the grid, so the transform alone does all the placing and no mesh is ever
    // rebuilt while dragging. A height of zero is the flat case: a quad has no
    // extent along local y, so that axis stays at 1 rather than collapsing the
    // matrix. A negative height builds the box below the grid.
    static Transform GridBoxTransform( const EditorGrid & grid, Vec3 min, Vec3 max, f32 height ) {
        Transform transform = TransformDefault();
        transform.rotation = EditorGridRotation( grid );
        transform.position = EditorGridToWorld( grid, Vec3{ 0.5f * ( min.x + max.x ),
                                                            0.5f * ( min.y + max.y ),
                                                            0.5f * height } );
        transform.scale = Vec3{ max.x - min.x,
                                height == 0.0f ? 1.0f : fabsf( height ),
                                max.y - min.y };
        return transform;
    }

    VulkanView::VulkanView( Renderer * renderer )
        : renderer( renderer ), world( WorldCreate() ), started( false ), startFailed( false ),
          panes(), paneCount( 0 ), layout( PaneLayout_Split ), activePane( kPerspectivePane ),
          movement(), dragging( false ), dragPane( kPerspectivePane ),
          grid( EditorGridDefault() ),
          buildStage( BuildStage_Off ), buildPrimitive( kNoPrimitive ), buildPane( -1 ),
          buildStart(), buildMin(), buildMax(), buildHeight( 0.0f ),
          editPrimitive( kNoPrimitive ), editVertex( kHMNone ), editGeometryDirty( false ),
          gizmo( GizmoCreate() ),
          dragAnchor(), frameTimer() {
        setSurfaceType( QSurface::VulkanSurface );
        PanesInit( panes );
        SetLayout( PaneLayout_Single );
    }

    VulkanView::~VulkanView() {
        // Authoring data first: it is plain heap memory, and the GPU meshes it
        // was built into belong to the renderer being shut down below.
        WorldFree( world );

        // Runs before the QWindow base destructor, so Qt's surface is still
        // alive while the swapchain that references it is torn down.
        RendererShutdownDevice( renderer );
    }

    void VulkanView::SetLayout( PaneLayout next ) {
        // A drag is measured against the pane it started in. Letting one
        // survive a layout change would finish it against a rectangle that has
        // moved out from under it.
        EndDrag();
        CancelBuild();

        layout = next;
        switch( next ) {
            case PaneLayout_Single:
                paneCount = 1;
                PaneSetRect( panes[0], 0.0f, 0.0f, 1.0f, 1.0f );
                break;

            case PaneLayout_Quad:
                paneCount = 4;
                PaneSetRect( panes[0], 0.0f, 0.0f, 0.5f, 0.5f );
                PaneSetRect( panes[1], 0.5f, 0.0f, 0.5f, 0.5f );
                PaneSetRect( panes[2], 0.0f, 0.5f, 0.5f, 0.5f );
                PaneSetRect( panes[3], 0.5f, 0.5f, 0.5f, 0.5f );
                break;

            case PaneLayout_Split:
            default:
                layout = PaneLayout_Split;
                paneCount = 2;
                PaneSetRect( panes[0], 0.0f, 0.0f, 0.5f, 1.0f );
                PaneSetRect( panes[1], 0.5f, 0.0f, 0.5f, 1.0f );
                break;
        }

        // A pane that just went away must not keep a camera stuck in look or
        // pan mode for the next time the layout brings it back.
        for( i32 i = paneCount; i < kMaxRenderViews; i++ ) {
            panes[i].flyInput.looking = false;
            panes[i].orthoInput = {};
        }
        if( activePane >= paneCount ) {
            activePane = kPerspectivePane;
        }
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

    QRect VulkanView::PaneRect( i32 pane ) const {
        const EditorPane & p = panes[pane];
        const i32 left = (i32)( p.x * (f32)width() );
        const i32 top = (i32)( p.y * (f32)height() );
        // The far edges come from the next boundary rather than from a scaled
        // width, so neighbouring panes meet exactly instead of leaving a
        // one-pixel seam that belongs to nobody.
        const i32 right = (i32)( ( p.x + p.width ) * (f32)width() );
        const i32 bottom = (i32)( ( p.y + p.height ) * (f32)height() );
        return QRect( left, top, right - left, bottom - top );
    }

    i32 VulkanView::PaneAt( QPoint position ) const {
        for( i32 i = 0; i < paneCount; i++ ) {
            if( PaneRect( i ).contains( position ) ) {
                return i;
            }
        }
        // A position off the window entirely - which a drag past the edge
        // produces - keeps working the pane it was already working.
        return activePane;
    }

    void VulkanView::SetActivePane( i32 pane ) {
        if( pane < 0 || pane >= paneCount ) {
            return;
        }
        activePane = pane;
    }

    i32 VulkanView::MovementPane() const {
        if( panes[activePane].kind == PaneKind_Perspective ) {
            return activePane;
        }
        // Hovering an orthographic pane still flies the perspective one, which
        // is what keeps WASD from going dead over most of a quad layout.
        for( i32 i = 0; i < paneCount; i++ ) {
            if( panes[i].kind == PaneKind_Perspective ) {
                return i;
            }
        }
        return -1;
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
        const i32 movePane = MovementPane();

        RenderView views[kMaxRenderViews] = {};
        for( i32 i = 0; i < paneCount; i++ ) {
            EditorPane & pane = panes[i];

            // Each pane's projection uses its own pixel size, or every one of
            // them would be stretched by the whole surface's aspect.
            const i32 paneWidth = (i32)( pane.width * (f32)surfaceWidth );
            const i32 paneHeight = (i32)( pane.height * (f32)surfaceHeight );

            if( pane.kind == PaneKind_Perspective ) {
                const FlyCameraInput keys = i == movePane ? movement : FlyCameraInput{};
                FlyCameraUpdate( &pane.fly, FlyInputCombine( pane.flyInput, keys ), dt );

                // Deltas are per-frame: whatever the mouse did before this
                // update has been applied, so the next frame starts from zero.
                pane.flyInput.lookDeltaX = 0.0f;
                pane.flyInput.lookDeltaY = 0.0f;

                views[i].viewProjection = FlyCameraViewProjection( pane.fly, paneWidth, paneHeight );
            } else {
                // Logical height, not the pixel one: the pan deltas come from
                // Qt cursor positions, which are logical too, and mixing the
                // two would scale panning by the display's device pixel ratio.
                OrthoCameraUpdate( &pane.ortho, pane.orthoInput, (i32)( pane.height * (f32)height() ) );

                pane.orthoInput.panDeltaX = 0.0f;
                pane.orthoInput.panDeltaY = 0.0f;
                pane.orthoInput.zoomTicks = 0.0f;

                views[i].viewProjection = OrthoCameraViewProjection( pane.ortho, paneWidth, paneHeight );
            }

            views[i].x = pane.x;
            views[i].y = pane.y;
            views[i].width = pane.width;
            views[i].height = pane.height;
        }

        // However many times the vertex moved since the last frame, the mesh
        // and its cage are rebuilt once.
        if( editGeometryDirty ) {
            editGeometryDirty = false;
            WorldRebuildPrimitive( world, renderer, editPrimitive );
            RefreshEditOverlay();
        }

        UpdateGizmo();

        // Build mode is checked first: it is the mode a click is answering to
        // when both are somehow up.
        if( buildStage != BuildStage_Off ) {
            RendererSetBorder( renderer, true, kBuildBorderColor );
        } else if( editPrimitive != kNoPrimitive ) {
            RendererSetBorder( renderer, true, kEditBorderColor );
        } else {
            RendererSetBorder( renderer, false, Vec3{} );
        }

        // The drawn grid and the grid geometry snaps to are one grid, so the
        // renderer is told where it is every frame rather than at the moments
        // someone remembers to. CPU-only, the same as the views below.
        RendererSetGridTransform( renderer, EditorGridTransform( grid ) );

        RendererSetViews( renderer, views, paneCount );
        RendererDrawFrame( renderer );

        // Presenting is FIFO, so this self-scheduling loop paces itself on vsync
        // instead of spinning the Qt event loop.
        requestUpdate();
    }

    void VulkanView::SetMovementKey( int key, bool pressed ) {
        switch( key ) {
            case Qt::Key_W:         movement.forward = pressed; break;
            case Qt::Key_S:         movement.back = pressed;    break;
            case Qt::Key_D:         movement.right = pressed;   break;
            case Qt::Key_A:         movement.left = pressed;    break;
            case Qt::Key_Space:     movement.up = pressed;      break;
            case Qt::Key_Control:   movement.down = pressed;    break;
            case Qt::Key_Shift:     movement.fast = pressed;    break;
            default: break;
        }
    }

    void VulkanView::BeginDrag( i32 pane ) {
        if( dragging || pane < 0 || pane >= paneCount ) {
            return;
        }
        dragging = true;
        dragPane = pane;
        SetActivePane( pane );

        // Anchor where the drag started and hide the pointer, so the cursor
        // does not wander into another pane or hit a screen edge mid-drag.
        dragAnchor = QCursor::pos();
        setCursor( Qt::BlankCursor );

        if( panes[pane].kind == PaneKind_Perspective ) {
            panes[pane].flyInput.looking = true;
        } else {
            panes[pane].orthoInput.panning = true;
        }
    }

    void VulkanView::EndDrag() {
        if( !dragging ) {
            return;
        }
        dragging = false;
        // Cleared across every slot rather than just the dragged one, so a
        // layout change during a drag cannot strand a camera in look or pan
        // mode with no press left to end it.
        for( i32 i = 0; i < kMaxRenderViews; i++ ) {
            panes[i].flyInput.looking = false;
            panes[i].orthoInput.panning = false;
        }

        unsetCursor();
        QCursor::setPos( dragAnchor );
    }

    void VulkanView::RayAt( QPoint position, i32 pane, Vec3 * outOrigin, Vec3 * outDirection ) const {
        const QRect rect = PaneRect( pane );
        const f32 localX = (f32)( position.x() - rect.x() );
        const f32 localY = (f32)( position.y() - rect.y() );

        if( panes[pane].kind == PaneKind_Perspective ) {
            FlyCameraScreenRay( panes[pane].fly, localX, localY,
                                rect.width(), rect.height(), outOrigin, outDirection );
        } else {
            OrthoCameraScreenRay( panes[pane].ortho, localX, localY,
                                  rect.width(), rect.height(), outOrigin, outDirection );
        }
    }

    bool VulkanView::PickAt( QPoint position, i32 pane, i32 * outPrimitive ) const {
        Vec3 origin = {};
        Vec3 direction = {};
        RayAt( position, pane, &origin, &direction );
        return WorldPick( world, renderer, origin, direction, outPrimitive );
    }

    Mat4 VulkanView::PaneViewProjection( i32 pane ) const {
        // Logical pixels are enough here: only the aspect ratio reaches the
        // projection, and that is the same either side of the pixel ratio.
        const QRect rect = PaneRect( pane );
        if( panes[pane].kind == PaneKind_Perspective ) {
            return FlyCameraViewProjection( panes[pane].fly, rect.width(), rect.height() );
        }
        return OrthoCameraViewProjection( panes[pane].ortho, rect.width(), rect.height() );
    }

    bool VulkanView::PickVertexAt( QPoint position, i32 pane, i32 * outVertex ) const {
        if( editPrimitive == kNoPrimitive ) {
            return false;
        }

        const QRect rect = PaneRect( pane );
        const f32 paneX = (f32)rect.x();
        const f32 paneY = (f32)rect.y();
        const f32 paneWidth = (f32)rect.width();
        const f32 paneHeight = (f32)rect.height();
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
            const f32 screenY = paneY + ( 0.5f - clip.y / clip.w * 0.5f ) * paneHeight;
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
            // mostly handled. Every orthographic pane draws it at that same
            // world size rather than one of its own.
            gizmo.scale = GizmoScaleFor( gizmo.center, panes[kPerspectivePane].fly.position );
        }

        RenderGizmoRange ranges[kGizmoRangeCount] = {};
        const i32 rangeCount = GizmoDrawRanges( gizmo, ranges );
        RendererSetGizmoDraw( renderer, GizmoDrawTransform( gizmo ), ranges, rangeCount );
    }

    bool VulkanView::BeginGizmoDrag( QPoint position, i32 pane ) {
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

    bool VulkanView::GridPointAt( QPoint position, i32 pane, Vec3 * outLocal ) const {
        if( pane < 0 || pane >= paneCount ) {
            return false;
        }

        Vec3 origin = {};
        Vec3 direction = {};
        RayAt( position, pane, &origin, &direction );
        return EditorGridRaycast( grid, origin, direction, outLocal );
    }

    void VulkanView::CycleGridPlane() {
        if( !started ) {
            return;
        }

        // Anything half drawn was drawn on the old plane, and would finish on
        // the new one.
        CancelBuild();

        // A normal that is not one of the three - nothing sets one today - is
        // read as the first, so the key always lands somewhere known.
        i32 current = 0;
        for( i32 i = 0; i < kGridNormalCount; i++ ) {
            if( Vec3Dot( grid.normal, kGridNormals[i] ) > 0.99f ) {
                current = i;
                break;
            }
        }

        grid.normal = kGridNormals[( current + 1 ) % kGridNormalCount];
    }

    bool VulkanView::AlignGridToFaceAt( QPoint position, i32 pane ) {
        if( !started || pane < 0 || pane >= paneCount ) {
            return false;
        }

        Vec3 origin = {};
        Vec3 direction = {};
        RayAt( position, pane, &origin, &direction );

        i32 primitive = kNoPrimitive;
        i32 face = kHMNone;
        Vec3 point = {};
        Vec3 normal = {};
        if( !WorldPickFace( world, renderer, origin, direction, &primitive, &face, &point, &normal ) ) {
            return false;
        }

        // Same reason as CycleGridPlane: a half drawn box was drawn on the old
        // plane and would be finished on the new one.
        CancelBuild();

        grid.normal = normal;
        // The face itself, not a snapped point: the grid lines are what get
        // snapped to, and they have to start on the surface the user pointed
        // at or the first thing built floats off it.
        grid.position = point;
        return true;
    }

    void VulkanView::ToggleBuildMode() {
        if( !started ) {
            return;
        }

        if( buildStage == BuildStage_Off ) {
            // Build mode takes the left button whole. Anything the pointer was
            // already part way through has to end here rather than run on
            // underneath a mode that will never send it another event.
            GizmoEndDrag( gizmo );
            // The two modes answer the same clicks in different ways, so only
            // one of them is ever up: entering build mode drops edit mode.
            if( editPrimitive != kNoPrimitive ) {
                ToggleEditMode();
            }
            buildStage = BuildStage_Ready;
        } else {
            CancelBuild();
            buildStage = BuildStage_Off;
        }
    }

    bool VulkanView::BuildMousePress( QPoint position, i32 pane ) {
        if( buildStage == BuildStage_Off || !started ) {
            return false;
        }

        // The click that ends an extrude. Taken on the press rather than the
        // release, so the box is finished the moment you commit to it and the
        // release that follows lands on a Ready stage with nothing to do.
        if( buildStage == BuildStage_Height ) {
            CommitBuild();
            return true;
        }

        // Mid-base, a second button is not a second box. Swallowed all the
        // same: in build mode a left click is never a selection.
        if( buildStage != BuildStage_Ready ) {
            return true;
        }

        Vec3 local = {};
        if( !GridPointAt( position, pane, &local ) ) {
            return true;
        }

        buildPane = pane;
        buildStart = EditorGridSnapLocal( grid, local );
        buildHeight = 0.0f;
        GridRect( grid, buildStart, buildStart, &buildMin, &buildMax );

        // The base starts as a quad because that is what it is - a flat
        // rectangle on the grid. It only becomes a box when the release below
        // says the base is done.
        HalfMesh quad = {};
        HalfMeshCreateQuad( quad, 1.0f );

        RenderMaterial material = RenderMaterialDefault();
        material.albedo = Vec3{ 0.45f, 0.62f, 0.50f };

        buildPrimitive = WorldAddPrimitive( world, renderer, quad, material, TransformDefault() );
        HalfMeshFree( quad );

        if( buildPrimitive == kNoPrimitive ) {
            fprintf( stderr, "Failed to start a build\n" );
            return true;
        }

        WorldSetSelected( world, renderer, buildPrimitive );
        buildStage = BuildStage_Base;
        ApplyBuildTransform();
        return true;
    }

    void VulkanView::BuildMouseMove( QPoint position ) {
        if( buildPrimitive == kNoPrimitive ) {
            return;
        }

        if( buildStage == BuildStage_Base ) {
            // The cursor leaving the grid - dragging past the horizon in a
            // perspective pane - holds the last good rectangle rather than
            // collapsing the base to nothing.
            Vec3 local = {};
            if( !GridPointAt( position, buildPane, &local ) ) {
                return;
            }
            GridRect( grid, buildStart, EditorGridSnapLocal( grid, local ), &buildMin, &buildMax );
            ApplyBuildTransform();
            return;
        }

        if( buildStage == BuildStage_Height ) {
            // An extrude is not on the grid any more, so it is measured against
            // the line the box is growing along instead: the normal through the
            // middle of the locked base.
            const Vec3 base = EditorGridToWorld( grid, Vec3{ 0.5f * ( buildMin.x + buildMax.x ),
                                                             0.5f * ( buildMin.y + buildMax.y ),
                                                             0.0f } );
            Vec3 origin = {};
            Vec3 direction = {};
            RayAt( position, buildPane, &origin, &direction );

            f32 height = 0.0f;
            // A pane looking straight down that line - the top view of a flat
            // grid - has no height in it at all. The last one stands and the
            // click still commits, which beats snapping the box to zero.
            if( !EditorGridRaycastHeight( grid, base, origin, direction, &height ) ) {
                return;
            }

            height = SnapTo( height, grid.step );
            if( fabsf( height ) < grid.step ) {
                // One cell is the smallest box, the same floor the base has.
                // Which way it points follows the drag, and only falls back to
                // the current side when the drag is sitting exactly on zero.
                const bool below = height < 0.0f || ( height == 0.0f && buildHeight < 0.0f );
                height = below ? -grid.step : grid.step;
            }

            buildHeight = height;
            ApplyBuildTransform();
        }
    }

    void VulkanView::BuildMouseRelease() {
        // Only the base drag ends on a button release. The extrude ends on the
        // next press, so the release that locked the base cannot also finish it.
        if( buildStage != BuildStage_Base ) {
            return;
        }

        if( buildPrimitive == kNoPrimitive ) {
            CancelBuild();
            return;
        }

        BuildToBox();
        // Opens at one cell rather than at nothing, so the box reads as a box
        // from the first frame of the extrude.
        buildHeight = grid.step;
        ApplyBuildTransform();
        buildStage = BuildStage_Height;
    }

    void VulkanView::ApplyBuildTransform() {
        if( buildPrimitive == kNoPrimitive ) {
            return;
        }
        WorldSetPrimitiveTransform( world, renderer, buildPrimitive,
                                    GridBoxTransform( grid, buildMin, buildMax, buildHeight ) );
    }

    void VulkanView::BuildToBox() {
        if( buildPrimitive == kNoPrimitive ) {
            return;
        }

        // Swapping the authored mesh under the primitive, rather than deleting
        // it and adding another, keeps its index and the selection pointing at
        // the same thing across the change.
        Primitive & primitive = world.primitives[buildPrimitive];
        HalfMeshFree( primitive.halfMesh );
        primitive.halfMesh = {};
        HalfMeshCreateCube( primitive.halfMesh, 1 );

        // Idles the device, which is why this is on the release that locks the
        // base and not on every mouse move.
        WorldRebuildPrimitive( world, renderer, buildPrimitive );
    }

    void VulkanView::CommitBuild() {
        // What was built stays, and stays selected. The mode does not: it drops
        // back to Ready so one B gets you as many boxes as you want.
        buildPrimitive = kNoPrimitive;
        buildPane = -1;
        buildHeight = 0.0f;
        buildStage = BuildStage_Ready;
    }

    void VulkanView::CancelBuild() {
        if( buildPrimitive != kNoPrimitive ) {
            const i32 removed = buildPrimitive;
            buildPrimitive = kNoPrimitive;
            if( WorldRemovePrimitive( world, renderer, removed ) ) {
                // Same bookkeeping a delete does: removal shifts every index
                // above it, and anything still holding one has to follow.
                editPrimitive = WorldRemapPrimitive( editPrimitive, removed );
                RefreshEditOverlay();
            }
        }

        buildPane = -1;
        buildHeight = 0.0f;
        // A cancel mid-box drops to Ready, not out of the mode: one bad drag
        // should not put the tool away.
        if( buildStage != BuildStage_Off ) {
            buildStage = BuildStage_Ready;
        }
    }

    void VulkanView::DeleteSelected() {
        if( !started || world.selected == kNoPrimitive ) {
            return;
        }

        const i32 removed = world.selected;
        if( !WorldRemovePrimitive( world, renderer, removed ) ) {
            return;
        }

        // A box being built right now is a primitive like any other, so the
        // index tracking it has to follow the removal. If it was the box itself
        // that went, the build has nothing left to shape and goes back to Ready.
        buildPrimitive = WorldRemapPrimitive( buildPrimitive, removed );
        if( buildPrimitive == kNoPrimitive ) {
            CancelBuild();
        }

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
            // Exclusive with build mode, same as the other direction: whatever
            // box was part way through is thrown away and the mode comes down.
            // First, because the cancel can take the selection with it.
            if( buildStage != BuildStage_Off ) {
                CancelBuild();
                buildStage = BuildStage_Off;
            }
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
                // The drawn grid and the grid geometry snaps to are the same
                // grid, so neither is allowed to change without the other.
                grid.step = kGridSteps[step];
                RendererSetGridSpacing( renderer, grid.step );
            }

            // Layouts sit on the function keys because the number row already
            // belongs to the grid sizes.
            if( event->key() == Qt::Key_F1 ) {
                SetLayout( PaneLayout_Single );
            }

            if( event->key() == Qt::Key_F2 ) {
                SetLayout( PaneLayout_Split );
            }

            if( event->key() == Qt::Key_F3 ) {
                SetLayout( PaneLayout_Quad );
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

            if( event->key() == Qt::Key_B ) {
                ToggleBuildMode();
            }

            if( event->key() == Qt::Key_G ) {
                CycleGridPlane();
            }

            // Escape backs out one step: it throws away a box part way through
            // and leaves the mode up, so the next drag starts clean.
            if( event->key() == Qt::Key_Escape && buildStage != BuildStage_Off ) {
                CancelBuild();
            }
        }

        // Tab is edit mode's key here, but it is also Qt's focus-navigation
        // key. Handing it to the base class moves focus to the next widget and
        // takes every later key with it, which reads as the viewport going
        // deaf rather than as focus having moved, so it stops here.
        if( event->key() == Qt::Key_Tab || event->key() == Qt::Key_Backtab ) {
            event->accept();
            return;
        }

        QWindow::keyPressEvent( event );
    }

    void VulkanView::keyReleaseEvent( QKeyEvent * event ) {
        if( !event->isAutoRepeat() ) {
            SetMovementKey( event->key(), false );
        }

        // Same reason as the press: the release of a focus key is a second
        // chance for Qt to navigate on it.
        if( event->key() == Qt::Key_Tab || event->key() == Qt::Key_Backtab ) {
            event->accept();
            return;
        }

        QWindow::keyReleaseEvent( event );
    }

    void VulkanView::mousePressEvent( QMouseEvent * event ) {
        // Keys only reach a QWindow that holds focus, and clicking the
        // viewport is how the user expects to hand it over.
        requestActivate();

        const QPoint position = event->position().toPoint();
        const i32 pane = PaneAt( position );
        SetActivePane( pane );

        if( event->button() == Qt::RightButton ) {
            BeginDrag( pane );
        } else if( event->button() == Qt::LeftButton ) {
            // Alt is read before every mode below, build mode included: aiming
            // the grid is a thing you do in the middle of building on it, so it
            // cannot be a click that any mode gets to answer instead.
            if( event->modifiers() & Qt::AltModifier ) {
                AlignGridToFaceAt( position, pane );
                QWindow::mousePressEvent( event );
                return;
            }

            // Build mode owns the left button outright: while it is up, a press
            // draws or finishes a box and never selects, places or grabs a
            // handle. B is the way back out.
            if( BuildMousePress( position, pane ) ) {
                QWindow::mousePressEvent( event );
                return;
            }

            // A press on a handle is a drag of the selection, never a pick of
            // whatever happens to lie behind it.
            if( BeginGizmoDrag( position, pane ) ) {
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
                PickVertexAt( position, pane, &vertex );
                if( vertex != editVertex ) {
                    editVertex = vertex;
                    RefreshEditOverlay();
                }
            } else {
                // Outside build mode a left click only ever selects. A click
                // that lands on something picks it, and empty space clears the
                // selection; placing geometry is build mode's job.
                i32 hit = kNoPrimitive;
                if( PickAt( position, pane, &hit ) ) {
                    WorldSetSelected( world, renderer, hit );
                } else {
                    WorldSetSelected( world, renderer, kNoPrimitive );
                }
            }
        }
        QWindow::mousePressEvent( event );
    }

    void VulkanView::mouseReleaseEvent( QMouseEvent * event ) {
        if( event->button() == Qt::RightButton ) {
            EndDrag();
        } else if( event->button() == Qt::LeftButton ) {
            BuildMouseRelease();
            GizmoEndDrag( gizmo );
        }
        QWindow::mouseReleaseEvent( event );
    }

    void VulkanView::mouseMoveEvent( QMouseEvent * event ) {
        const QPoint position = event->position().toPoint();

        // The pane under the cursor takes the keyboard, except while a drag is
        // warping the pointer back to its anchor, where the reported position
        // says nothing about what the user is pointing at.
        if( !dragging ) {
            SetActivePane( PaneAt( position ) );
        }

        // A camera drag is warping the pointer back to an anchor, so the
        // reported position says nothing about where a box should go.
        if( !dragging ) {
            BuildMouseMove( position );
        }

        if( gizmo.mode != GizmoMode_None && !dragging ) {
            Vec3 origin = {};
            Vec3 direction = {};
            RayAt( position, activePane, &origin, &direction );

            if( gizmo.active != GizmoAxis_None ) {
                Transform transform = {};
                // Ctrl turns the rotation snap off for fine adjustment.
                const f32 rotateSnap = ( event->modifiers() & Qt::ControlModifier ) ? 0.0f : kGizmoRotateSnap;
                if( GizmoUpdateDrag( gizmo, origin, direction, renderer->gridSpacing, rotateSnap, &transform ) ) {
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

        if( dragging ) {
            const QPoint global = event->globalPosition().toPoint();
            const QPoint delta = global - dragAnchor;
            // The warp below generates its own move event landing exactly on
            // the anchor; ignoring a zero delta is what stops it recursing.
            if( !delta.isNull() ) {
                EditorPane & pane = panes[dragPane];
                if( pane.kind == PaneKind_Perspective ) {
                    pane.flyInput.lookDeltaX += (f32)delta.x();
                    pane.flyInput.lookDeltaY += (f32)delta.y();
                } else {
                    pane.orthoInput.panDeltaX += (f32)delta.x();
                    pane.orthoInput.panDeltaY += (f32)delta.y();
                }
                QCursor::setPos( dragAnchor );
            }
        }
        QWindow::mouseMoveEvent( event );
    }

    void VulkanView::wheelEvent( QWheelEvent * event ) {
        // Zoom belongs to the pane under the cursor, and only an orthographic
        // one has a zoom to speak of.
        const i32 pane = PaneAt( event->position().toPoint() );
        if( panes[pane].kind == PaneKind_Ortho ) {
            // A notch is 120 eighths of a degree by Qt's convention.
            panes[pane].orthoInput.zoomTicks += (f32)event->angleDelta().y() / 120.0f;
        }
        QWindow::wheelEvent( event );
    }

    void VulkanView::focusOutEvent( QFocusEvent * event ) {
        // Releases arrive at whoever has focus, so a key or button still down
        // when focus leaves would otherwise stick on forever.
        EndDrag();
        // A base drag needs the button that is no longer being watched, so the
        // half-built base goes rather than sitting there waiting for a release
        // that will never arrive. An extrude is driven by moves and a click, so
        // it survives and picks up where it was.
        if( buildStage == BuildStage_Base ) {
            CancelBuild();
        }
        GizmoEndDrag( gizmo );
        movement = {};
        for( i32 i = 0; i < kMaxRenderViews; i++ ) {
            panes[i].flyInput = {};
            panes[i].orthoInput = {};
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
