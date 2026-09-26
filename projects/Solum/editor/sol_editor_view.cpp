#include "sol_editor_view.h"

#include <QClipboard>
#include <QCursor>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFocusEvent>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPlatformSurfaceEvent>
#include <QResizeEvent>
#include <QVulkanInstance>
#include <QWheelEvent>

#include <cmath>
#include <cstdio>

namespace sol {

    // The quad layout is the widest this goes, and it is exactly what the
    // renderer will draw in one frame.
    static_assert( kMaxRenderViews >= 4, "the editor lays out four panes" );

    // Selectable grid sizes, smallest first, on the 1-8 keys and [ ]. The snap
    // step follows whichever is current, so the grid is not decoration - it is
    // what geometry lands on.
    constexpr f32 kGridSteps[] = { 0.0625f, 0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f };
    constexpr i32 kGridStepCount = (i32)SPLATS_ARRAY_COUNT( kGridSteps );
    constexpr i32 kDefaultGridStep = 3;

    // Slot 0 is the perspective pane in every layout; the rest are 2D. Slot
    // indices are also render view indices, which is what lets a batch pick
    // its panes with a fixed bit mask.
    constexpr i32 kPerspectivePane = 0;
    constexpr u32 kMask3D = 1u << kPerspectivePane;
    constexpr u32 kMask2D = ~kMask3D & ( ( 1u << kMaxRenderViews ) - 1u );

    // The frame each pane wears. The pane the keyboard goes to wears the
    // current tool's colour, which is the one place the mode is always shown.
    constexpr Vec3 kPaneFrameColor = { 0.16f, 0.16f, 0.19f };
    constexpr Vec3 kToolColors[EditorTool_Count] = {
        { 0.35f, 0.58f, 0.95f },    // brush
        { 0.98f, 0.62f, 0.20f },    // clip
        { 0.30f, 0.85f, 0.42f },    // vertex
        { 0.30f, 0.85f, 0.42f },    // edge
        { 0.30f, 0.85f, 0.42f },    // face
        { 0.72f, 0.48f, 0.95f },    // rotate
    };

    constexpr f32 kPitchLimit = 89.0f * kDeg2Rad;

    const char * EditorToolName( EditorTool tool ) {
        switch( tool ) {
            case EditorTool_Brush:  return "Brush";
            case EditorTool_Clip:   return "Clip";
            case EditorTool_Vertex: return "Vertex";
            case EditorTool_Edge:   return "Edge";
            case EditorTool_Face:   return "Face";
            case EditorTool_Rotate: return "Rotate";
            default:                return "?";
        }
    }

    // Builds the cameras for all four slots once. A layout change rewrites
    // rectangles only, so a camera framed up in one layout is still pointing
    // the same way when that layout comes back.
    static void PanesInit( EditorPane * panes ) {
        panes[0] = {};
        panes[0].kind = PaneKind_Perspective;
        panes[0].fly = FlyCameraDefault();
        panes[0].fly.moveSpeed = 6.0f;
        // Looking down on the level from above a corner, the way a level is
        // mostly worked on; framing then only has to back the camera off.
        panes[0].fly.yaw = -0.62f;
        panes[0].fly.pitch = -0.5f;

        const OrthoAxis axes[3] = { OrthoAxis_Top, OrthoAxis_Front, OrthoAxis_Side };
        for( i32 i = 1; i < 4; i++ ) {
            panes[i] = {};
            panes[i].kind = PaneKind_Ortho;
            panes[i].ortho = OrthoCameraDefault( axes[i - 1] );
            panes[i].ortho.halfHeight = 8.0f;
        }
    }

    static void PaneSetRect( EditorPane & pane, f32 x, f32 y, f32 width, f32 height ) {
        pane.x = x;
        pane.y = y;
        pane.width = width;
        pane.height = height;
    }

    VulkanView::VulkanView( Renderer * renderer )
        : renderer( renderer ), started( false ), startFailed( false ), dialogParent( nullptr ) {
        setSurfaceType( QSurface::VulkanSurface );

        doc = DocCreate();
        textures = {};
        worldStream = {};
        backgroundStream = {};
        overlayStream = {};
        builtWorldVersion = 0;
        previewVersion = 0;
        builtPreviewVersion = 0;

        PanesInit( panes );
        layout = PaneLayout_Quad;
        maximizedPane = -1;
        activePane = kPerspectivePane;
        movement = {};

        cameraDrag = CameraDrag_None;
        cameraButton = Qt::NoButton;
        cameraPane = kPerspectivePane;
        orbitPivot = {};

        grid = EditorGridDefault();
        gridStepIndex = kDefaultGridStep;
        grid.step = kGridSteps[gridStepIndex];
        textureLock = true;

        tool = EditorTool_Brush;
        currentTexture = FaceTextureDefault();

        leftDrag = LeftDrag_None;
        pendingDrag = LeftDrag_None;
        pressPane = kPerspectivePane;
        pressModifiers = Qt::NoModifier;
        pressPick = {};
        pressPick.brush = kNoBrush;

        dragIndices = {};
        dragOriginals = {};
        dragBoundsMin = {};
        dragBoundsMax = {};
        dragDelta = {};
        dragChanged = false;

        moveBase = {};
        moveAnchor = {};
        movePhaseHit = {};
        moveAnchorT = 0.0f;
        moveVertical = false;
        movePhaseStarted = false;

        createBrush = {};
        createValid = false;
        createAxis = 1;
        createSign = 1.0f;
        createPlane = 0.0f;
        createStart = {};
        createEnd = {};
        createHeight = 0.0f;
        createHeightMode = false;
        createHeightAnchorT = 0.0f;
        createHeightAtAnchor = 0.0f;
        referenceMin = Vec3{ 0.0f, 0.0f, 0.0f };
        referenceMax = Vec3{ 1.0f, 2.0f, 1.0f };

        faceTargetCount = 0;
        faceAmounts[0] = 0.0f;
        faceAmounts[1] = 0.0f;
        extrudeBrush = {};
        extrudeValid = false;
        hoverTargetCount = 0;
        hoverPane = -1;
        hoverCursor = Qt::ArrowCursor;

        handleSelection = {};
        handleDragStart = {};
        handleVertexStart = {};
        handleVertices = {};
        handleAffected = {};

        clipPointCount = 0;
        clipDragPoint = -1;
        clipSide = ClipSide_Back;
        for( i32 i = 0; i < kMaxClipPoints; i++ ) {
            clipPoints[i] = {};
            clipPointAxis[i] = {};
            clipPointNormal[i] = {};
        }

        gizmo = GizmoCreate();
        rotateAngle = 0.0f;

        duplicateActive = false;
        duplicateStride = {};
        duplicateSourceMin = {};

        lastFieldEdit = -1;
        lastFieldEditVersion = 0;

        SetLayout( PaneLayout_Quad );
        CreateStarterMap();
    }

    VulkanView::~VulkanView() {
        FreeSnapshot();
        BrushFree( createBrush );
        BrushFree( extrudeBrush );
        ListFree( handleSelection );
        ListFree( handleVertexStart );
        ListFree( handleVertices );
        ListFree( handleAffected );
        StreamFree( worldStream );
        StreamFree( backgroundStream );
        StreamFree( overlayStream );
        EditorTexturesFree( textures );
        DocFree( doc );

        // Normally already done when the surface went away (see event()).
        // This covers a view destroyed without a container, and is a no-op
        // on a device that is already down.
        RendererShutdownDevice( renderer );
    }

    void VulkanView::SetLayout( PaneLayout next ) {
        // A drag is measured against the pane it started in. Letting one
        // survive a layout change would finish it against a rectangle that has
        // moved out from under it.
        CancelDrag();
        EndCameraDrag();

        if( next != layout ) {
            maximizedPane = -1;
        }
        layout = next;

        for( i32 i = 0; i < kMaxRenderViews; i++ ) {
            PaneSetRect( panes[i], 0.0f, 0.0f, 0.0f, 0.0f );
        }

        switch( layout ) {
            case PaneLayout_Single:
                PaneSetRect( panes[0], 0.0f, 0.0f, 1.0f, 1.0f );
                break;
            case PaneLayout_Split:
                PaneSetRect( panes[0], 0.0f, 0.0f, 0.5f, 1.0f );
                PaneSetRect( panes[1], 0.5f, 0.0f, 0.5f, 1.0f );
                break;
            case PaneLayout_Tall: {
                const f32 split = 0.64f;
                const f32 third = 1.0f / 3.0f;
                PaneSetRect( panes[0], 0.0f, 0.0f, split, 1.0f );
                PaneSetRect( panes[1], split, 0.0f, 1.0f - split, third );
                PaneSetRect( panes[2], split, third, 1.0f - split, third );
                PaneSetRect( panes[3], split, 2.0f * third, 1.0f - split, 1.0f - 2.0f * third );
                break;
            }
            case PaneLayout_Quad:
            default:
                layout = PaneLayout_Quad;
                PaneSetRect( panes[0], 0.0f, 0.0f, 0.5f, 0.5f );
                PaneSetRect( panes[1], 0.5f, 0.0f, 0.5f, 0.5f );
                PaneSetRect( panes[2], 0.0f, 0.5f, 0.5f, 0.5f );
                PaneSetRect( panes[3], 0.5f, 0.5f, 0.5f, 0.5f );
                break;
        }

        // Maximising shows one pane of the layout over the whole window and
        // collapses the rest; the layout comes back untouched after.
        if( maximizedPane >= 0 && PaneVisible( maximizedPane ) ) {
            for( i32 i = 0; i < kMaxRenderViews; i++ ) {
                if( i != maximizedPane ) {
                    PaneSetRect( panes[i], 0.0f, 0.0f, 0.0f, 0.0f );
                }
            }
            PaneSetRect( panes[maximizedPane], 0.0f, 0.0f, 1.0f, 1.0f );
        } else {
            maximizedPane = -1;
        }

        for( i32 i = 0; i < kMaxRenderViews; i++ ) {
            panes[i].flyInput.looking = false;
            panes[i].orthoInput = {};
        }
        // The keyboard has to go somewhere that can be seen; with one pane
        // maximised that may not be the perspective one.
        for( i32 i = 0; i < kMaxRenderViews && !PaneVisible( activePane ); i++ ) {
            activePane = i;
        }
        hoverTargetCount = 0;
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

        if( !EditorTexturesInit( textures, renderer ) ) {
            fprintf( stderr, "Failed to build the dev texture\n" );
        }

        RendererSetGridSpacing( renderer, grid.step );

        // Local-space geometry that never changes, so it is uploaded once and
        // then only ever repositioned by a push constant.
        List<StaticMeshVertex> gizmoVertices = {};
        GizmoBuildGeometry( gizmo, gizmoVertices );
        RendererSetGizmoGeometry( renderer, gizmoVertices.data, gizmoVertices.count );
        ListFree( gizmoVertices );

        started = true;

        Vec3 min = {};
        Vec3 max = {};
        if( DocVisibleBounds( doc, &min, &max ) ) {
            FrameBounds( min, max );
        }
        return true;
    }

    // --- panes --------------------------------------------------------------

    bool VulkanView::PaneVisible( i32 pane ) const {
        return pane >= 0 && pane < kMaxRenderViews && panes[pane].width > 0.0f && panes[pane].height > 0.0f;
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
        for( i32 i = 0; i < kMaxRenderViews; i++ ) {
            if( PaneVisible( i ) && PaneRect( i ).contains( position ) ) {
                return i;
            }
        }
        // A position off the window entirely - which a drag past the edge
        // produces - keeps working the pane it was already working.
        return activePane;
    }

    bool VulkanView::PaneIs2D( i32 pane ) const {
        return panes[pane].kind == PaneKind_Ortho;
    }

    void VulkanView::PaneAxes( i32 pane, Vec3 * outRight, Vec3 * outUp, Vec3 * outForward ) const {
        if( PaneIs2D( pane ) ) {
            OrthoCameraAxes( panes[pane].ortho, outRight, outUp, outForward );
        } else {
            FlyCameraAxes( panes[pane].fly, outRight, outUp, outForward );
        }
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

    void VulkanView::RayAt( QPoint position, i32 pane, Vec3 * outOrigin, Vec3 * outDirection ) const {
        const QRect rect = PaneRect( pane );
        const f32 localX = (f32)( position.x() - rect.x() );
        const f32 localY = (f32)( position.y() - rect.y() );

        if( panes[pane].kind == PaneKind_Perspective ) {
            FlyCameraScreenRay( panes[pane].fly, localX, localY, rect.width(), rect.height(), outOrigin, outDirection );
        } else {
            OrthoCameraScreenRay( panes[pane].ortho, localX, localY, rect.width(), rect.height(), outOrigin, outDirection );
        }
    }

    bool VulkanView::WorldToScreen( i32 pane, Vec3 world, f32 * outX, f32 * outY ) const {
        const QRect rect = PaneRect( pane );
        const Vec4 clip = PaneViewProjection( pane ) * Vec4{ world.x, world.y, world.z, 1.0f };
        // Behind the camera, where the divide would mirror it back on screen.
        if( clip.w <= 1e-6f ) {
            return false;
        }
        // The renderer's negative viewport height puts clip +y at the top, so
        // screen y runs the other way from it.
        *outX = (f32)rect.x() + ( clip.x / clip.w * 0.5f + 0.5f ) * (f32)rect.width();
        *outY = (f32)rect.y() + ( 0.5f - clip.y / clip.w * 0.5f ) * (f32)rect.height();
        return true;
    }

    Vec3 VulkanView::PlanePoint2D( QPoint position, i32 pane ) const {
        const QRect rect = PaneRect( pane );
        return OrthoCameraScreenToWorld( panes[pane].ortho, (f32)( position.x() - rect.x() ), (f32)( position.y() - rect.y() ),
                                         rect.width(), rect.height() );
    }

    PickResult VulkanView::PickAt( QPoint position, i32 pane ) const {
        PickResult result = {};
        result.brush = kNoBrush;
        result.face = -1;

        Vec3 origin = {};
        Vec3 direction = {};
        RayAt( position, pane, &origin, &direction );

        f32 distance = 0.0f;
        if( DocPick( doc, origin, direction, &result.brush, &result.face, &distance ) ) {
            result.point = origin + direction * distance;
            result.normal = doc.map.brushes[result.brush].faces[result.face].plane.normal;
        }
        return result;
    }

    // --- frame --------------------------------------------------------------

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

        RenderView views[kMaxRenderViews] = {};
        for( i32 i = 0; i < kMaxRenderViews; i++ ) {
            EditorPane & pane = panes[i];

            // Each pane's projection uses its own pixel size, or every one of
            // them would be stretched by the whole surface's aspect.
            const i32 paneWidth = (i32)( pane.width * (f32)surfaceWidth );
            const i32 paneHeight = (i32)( pane.height * (f32)surfaceHeight );

            if( pane.kind == PaneKind_Perspective ) {
                // Keys fly the camera only while its right button is held, so
                // the rest of the time they belong to the tools.
                const bool flying = cameraDrag == CameraDrag_Look && cameraPane == i;
                FlyCameraInput input = flying ? movement : FlyCameraInput{};
                input.looking = pane.flyInput.looking;
                input.lookDeltaX = pane.flyInput.lookDeltaX;
                input.lookDeltaY = pane.flyInput.lookDeltaY;
                FlyCameraUpdate( &pane.fly, input, dt );
                pane.flyInput.lookDeltaX = 0.0f;
                pane.flyInput.lookDeltaY = 0.0f;
                views[i].viewProjection = FlyCameraViewProjection( pane.fly, paneWidth, paneHeight );
            } else {
                // Logical height, not the pixel one: the pan deltas come from
                // Qt cursor positions, which are logical too.
                OrthoCameraUpdate( &pane.ortho, pane.orthoInput, (i32)( pane.height * (f32)height() ) );
                pane.orthoInput.panDeltaX = 0.0f;
                pane.orthoInput.panDeltaY = 0.0f;
                pane.orthoInput.zoomTicks = 0.0f;
                views[i].viewProjection = OrthoCameraViewProjection( pane.ortho, paneWidth, paneHeight );
                views[i].hideGrid = true;
            }

            views[i].x = pane.x;
            views[i].y = pane.y;
            views[i].width = pane.width;
            views[i].height = pane.height;
        }

        // The world stream is only rebuilt when the document or a preview
        // changed; the rest of the time last frame's copy keeps drawing.
        if( doc.version != builtWorldVersion || previewVersion != builtPreviewVersion ) {
            Brush previews[2] = {};
            i32 previewCount = 0;
            if( leftDrag == LeftDrag_Create && createValid ) {
                previews[previewCount++] = createBrush;
            }
            if( leftDrag == LeftDrag_Extrude && extrudeValid ) {
                previews[previewCount++] = extrudeBrush;
            }
            EditorBuildWorld( worldStream, doc, previews, previewCount, textures, renderer, kMask3D, kMask2D );
            StreamSubmit( worldStream, renderer, RenderStream_World );
            builtWorldVersion = doc.version;
            builtPreviewVersion = previewVersion;
        }

        BuildBackground();
        BuildOverlay();
        UpdateRotateGizmo();

        RendererSetGridTransform( renderer, EditorGridTransform( grid ) );
        RendererSetViews( renderer, views, kMaxRenderViews );
        RendererDrawFrame( renderer );

        // Presenting is FIFO, so this self-scheduling loop paces itself on vsync
        // instead of spinning the Qt event loop.
        requestUpdate();
    }

    void VulkanView::BuildBackground() {
        const qreal dpr = devicePixelRatio();
        StreamClear( backgroundStream );
        for( i32 i = 0; i < kMaxRenderViews; i++ ) {
            if( !PaneVisible( i ) || !PaneIs2D( i ) ) {
                continue;
            }
            const QRect rect = PaneRect( i );
            EditorDrawOrthoGrid( backgroundStream, panes[i].ortho, (i32)( rect.width() * dpr ), (i32)( rect.height() * dpr ),
                                 grid.step, 1u << (u32)i );
        }
        StreamSubmit( backgroundStream, renderer, RenderStream_Background );
    }

    void VulkanView::BuildOverlay() {
        StreamClear( overlayStream );
        DrawToolOverlay( overlayStream );

        const qreal dpr = devicePixelRatio();
        for( i32 i = 0; i < kMaxRenderViews; i++ ) {
            if( !PaneVisible( i ) ) {
                continue;
            }
            const QRect rect = PaneRect( i );
            const f32 w = (f32)( rect.width() * dpr );
            const f32 h = (f32)( rect.height() * dpr );

            RenderBatch * batch = StreamBatch( overlayStream, RenderBatch_LinesOnTop, kNoTint, 1u << (u32)i );
            batch->screenSpace = true;

            Vec3 right = {};
            Vec3 up = {};
            Vec3 forward = {};
            PaneAxes( i, &right, &up, &forward );
            EditorDrawAxisTripod( overlayStream, right, up, w, h );

            // Pixel centres, so each ring of the frame is exactly one pixel.
            const bool active = i == activePane;
            const Vec3 color = active ? kToolColors[tool] : kPaneFrameColor;
            const i32 rings = active ? 2 : 1;
            for( i32 ring = 0; ring < rings; ring++ ) {
                const f32 inset = 0.5f + (f32)ring;
                EditorDrawScreenRect( overlayStream, inset, inset, w - inset, h - inset, w, h, color );
            }
        }

        StreamSubmit( overlayStream, renderer, RenderStream_Overlay );
    }

    void VulkanView::UpdateRotateGizmo() {
        Vec3 min = {};
        Vec3 max = {};
        const bool show = tool == EditorTool_Rotate && DocSelectionBounds( doc, &min, &max );
        RendererSetGizmoVisible( renderer, show );
        if( !show ) {
            if( leftDrag != LeftDrag_Rotate ) {
                GizmoEndDrag( gizmo );
            }
            gizmo.hovered = GizmoAxis_None;
            return;
        }

        gizmo.mode = GizmoMode_Rotate;
        // Not moved mid-drag: the rings have to stay where they were grabbed,
        // or turning the selection would move the circle the drag is measured
        // against.
        if( gizmo.active == GizmoAxis_None ) {
            gizmo.center = ( min + max ) * 0.5f;
            gizmo.scale = GizmoScaleFor( gizmo.center, panes[kPerspectivePane].fly.position );
        }

        RenderGizmoRange ranges[kGizmoRangeCount] = {};
        const i32 rangeCount = GizmoDrawRanges( gizmo, ranges );
        RendererSetGizmoDraw( renderer, GizmoDrawTransform( gizmo ), ranges, rangeCount );
    }

    // --- cameras ------------------------------------------------------------

    void VulkanView::SetMovementKey( int key, bool pressed ) {
        switch( key ) {
            case Qt::Key_W:     movement.forward = pressed; break;
            case Qt::Key_S:     movement.back = pressed;    break;
            case Qt::Key_D:     movement.right = pressed;   break;
            case Qt::Key_A:     movement.left = pressed;    break;
            case Qt::Key_E:     movement.up = pressed;      break;
            case Qt::Key_Q:     movement.down = pressed;    break;
            case Qt::Key_Shift: movement.fast = pressed;    break;
            default: break;
        }
    }

    void VulkanView::BeginCameraDrag( CameraDrag kind, i32 pane, QPoint position ) {
        if( cameraDrag != CameraDrag_None || !PaneVisible( pane ) ) {
            return;
        }
        cameraDrag = kind;
        cameraPane = pane;
        activePane = pane;
        hoverTargetCount = 0;

        // Anchor where the drag started and hide the pointer, so the cursor
        // does not wander into another pane or hit a screen edge mid-drag.
        cameraAnchor = QCursor::pos();
        setCursor( Qt::BlankCursor );

        if( kind == CameraDrag_Look ) {
            panes[pane].flyInput.looking = true;
        } else if( kind == CameraDrag_Pan && PaneIs2D( pane ) ) {
            panes[pane].orthoInput.panning = true;
        } else if( kind == CameraDrag_Orbit ) {
            // Round whatever is under the cursor, so the thing being looked at
            // stays put while the view swings about it.
            const PickResult pick = PickAt( position, pane );
            if( pick.brush != kNoBrush ) {
                orbitPivot = pick.point;
            } else {
                Vec3 origin = {};
                Vec3 direction = {};
                RayAt( position, pane, &origin, &direction );
                Vec3 local = {};
                if( EditorGridRaycast( grid, origin, direction, &local ) && Vec3Length( EditorGridToWorld( grid, local ) - origin ) < 200.0f ) {
                    orbitPivot = EditorGridToWorld( grid, local );
                } else {
                    orbitPivot = origin + direction * 8.0f;
                }
            }
        }
    }

    void VulkanView::EndCameraDrag() {
        if( cameraDrag == CameraDrag_None ) {
            return;
        }
        cameraDrag = CameraDrag_None;
        cameraButton = Qt::NoButton;
        // Cleared across every slot, so a layout change during a drag cannot
        // strand a camera in look or pan mode with no press left to end it.
        for( i32 i = 0; i < kMaxRenderViews; i++ ) {
            panes[i].flyInput.looking = false;
            panes[i].orthoInput.panning = false;
        }
        unsetCursor();
        // unsetCursor is the arrow, whatever hover last asked for.
        hoverCursor = Qt::ArrowCursor;
        QCursor::setPos( cameraAnchor );
    }

    void VulkanView::OrbitBy( f32 dx, f32 dy ) {
        FlyCamera & camera = panes[cameraPane].fly;

        // The offset to the pivot is held fixed in the camera's own frame, so
        // turning the frame carries the camera round the pivot and the pivot
        // stays on the same pixel.
        Vec3 right = {};
        Vec3 up = {};
        Vec3 forward = {};
        FlyCameraAxes( camera, &right, &up, &forward );
        const Vec3 offset = camera.position - orbitPivot;
        const f32 localRight = Vec3Dot( offset, right );
        const f32 localUp = Vec3Dot( offset, up );
        const f32 localForward = Vec3Dot( offset, forward );

        camera.yaw += dx * camera.lookSpeed;
        camera.pitch -= dy * camera.lookSpeed;
        if( camera.pitch > kPitchLimit )  { camera.pitch = kPitchLimit; }
        if( camera.pitch < -kPitchLimit ) { camera.pitch = -kPitchLimit; }

        FlyCameraAxes( camera, &right, &up, &forward );
        camera.position = orbitPivot + right * localRight + up * localUp + forward * localForward;
    }

    void VulkanView::PanPerspectiveBy( f32 dx, f32 dy ) {
        FlyCamera & camera = panes[cameraPane].fly;
        Vec3 right = {};
        Vec3 up = {};
        Vec3 forward = {};
        FlyCameraAxes( camera, &right, &up, &forward );
        // The world follows the cursor, the way a 2D pane pans.
        const f32 scale = camera.moveSpeed * 0.0035f;
        camera.position = camera.position - right * ( dx * scale ) + up * ( dy * scale );
    }

    void VulkanView::DollyAt( QPoint position, i32 pane, f32 notches ) {
        Vec3 origin = {};
        Vec3 direction = {};
        RayAt( position, pane, &origin, &direction );

        // Towards whatever is under the cursor, by a share of the distance to
        // it, so the wheel closes in on a point quickly from far off and gently
        // up close, and never steps through what it is heading for.
        f32 distance = 12.0f;
        const PickResult pick = PickAt( position, pane );
        if( pick.brush != kNoBrush ) {
            distance = Vec3Length( pick.point - origin );
        } else {
            Vec3 local = {};
            if( EditorGridRaycast( grid, origin, direction, &local ) ) {
                distance = Min( Vec3Length( EditorGridToWorld( grid, local ) - origin ), 200.0f );
            }
        }

        f32 step = Max( distance * 0.18f, 0.25f ) * notches;
        if( notches > 0.0f && step > distance - 0.2f ) {
            step = Max( distance - 0.2f, 0.0f );
        }
        panes[pane].fly.position = panes[pane].fly.position + direction * step;
    }

    void VulkanView::ZoomOrthoAt( QPoint position, i32 pane, f32 notches ) {
        OrthoCamera & camera = panes[pane].ortho;
        // Zooms about the cursor: the world point under it before the zoom is
        // put back under it after.
        const Vec3 before = PlanePoint2D( position, pane );
        camera.halfHeight *= powf( 1.18f, -notches );
        if( camera.halfHeight < 0.05f )   { camera.halfHeight = 0.05f; }
        if( camera.halfHeight > 2000.0f ) { camera.halfHeight = 2000.0f; }
        const Vec3 after = PlanePoint2D( position, pane );
        camera.center = camera.center + ( before - after );
    }

    void VulkanView::FrameBounds( Vec3 min, Vec3 max ) {
        const Vec3 center = ( min + max ) * 0.5f;
        const f32 radius = Max( Vec3Length( max - min ) * 0.5f, 1.0f );

        FlyCamera & fly = panes[kPerspectivePane].fly;
        const Vec3 forward = FlyCameraForward( fly );
        // Far enough back that a sphere round the bounds fits the 60 degree
        // field of view, with a little air.
        fly.position = center - forward * ( radius / tanf( 30.0f * kDeg2Rad ) * 1.1f );

        for( i32 i = 1; i < kMaxRenderViews; i++ ) {
            OrthoCamera & ortho = panes[i].ortho;
            Vec3 right = {};
            Vec3 up = {};
            Vec3 viewForward = {};
            OrthoCameraAxes( ortho, &right, &up, &viewForward );
            // Only the in-plane part of the centre moves: the depth the
            // camera sits at does not change what it shows.
            ortho.center = ortho.center + right * Vec3Dot( center - ortho.center, right ) + up * Vec3Dot( center - ortho.center, up );

            const QRect rect = PaneRect( i );
            const f32 aspect = rect.height() > 0 ? (f32)rect.width() / (f32)rect.height() : 1.0f;
            const Vec3 extent = max - min;
            const f32 extentRight = fabsf( Vec3Dot( extent, right ) );
            const f32 extentUp = fabsf( Vec3Dot( extent, up ) );
            ortho.halfHeight = Max( Max( extentUp, extentRight / Max( aspect, 0.1f ) ) * 0.6f, 1.0f );
        }
    }

    // --- keys ---------------------------------------------------------------

    void VulkanView::SetGridStepIndex( i32 index ) {
        if( index < 0 ) {
            index = 0;
        }
        if( index >= kGridStepCount ) {
            index = kGridStepCount - 1;
        }
        gridStepIndex = index;
        grid.step = kGridSteps[index];
        // The drawn grid and the grid geometry snaps to are the same grid, so
        // neither is allowed to change without the other. A stall, which is
        // fine on a key press.
        if( started ) {
            RendererSetGridSpacing( renderer, grid.step );
        }
        ShowMessage( QStringLiteral( "Grid %1" ).arg( (double)grid.step ) );
    }

    void VulkanView::SetTool( EditorTool next ) {
        CancelDrag();
        // The key of the tool already up puts it away, so every tool is a
        // toggle off the brush tool.
        if( next == tool && next != EditorTool_Brush ) {
            next = EditorTool_Brush;
        }
        if( next != EditorTool_Clip ) {
            clipPointCount = 0;
        }
        if( next != tool ) {
            ListClear( handleSelection );
        }
        tool = next;
        gizmo.mode = tool == EditorTool_Rotate ? GizmoMode_Rotate : GizmoMode_None;
        GizmoEndDrag( gizmo );
        hoverTargetCount = 0;
    }

    void VulkanView::ShowMessage( const QString & text ) {
        message = text;
        messageTimer.restart();
    }

    bool VulkanView::HandleKey( QKeyEvent * event ) {
        const int key = event->key();
        const Qt::KeyboardModifiers modifiers = event->modifiers();
        const bool ctrl = ( modifiers & Qt::ControlModifier ) != 0;
        const bool shift = ( modifiers & Qt::ShiftModifier ) != 0;
        const bool alt = ( modifiers & Qt::AltModifier ) != 0;
        const bool repeat = event->isAutoRepeat();

        // While the right button flies the camera the movement keys are the
        // camera's, and nothing else should fire off them.
        if( cameraDrag == CameraDrag_Look ) {
            switch( key ) {
                case Qt::Key_W: case Qt::Key_A: case Qt::Key_S: case Qt::Key_D:
                case Qt::Key_Q: case Qt::Key_E: case Qt::Key_Shift:
                    return true;
                default:
                    break;
            }
        }

        // Nudges repeat when held; everything else fires once per press.
        switch( key ) {
            case Qt::Key_Left:
                if( ctrl ) { if( !repeat ) Command( EditorCommand_RotateLeft ); } else { Nudge( -1, 0, 0 ); }
                return true;
            case Qt::Key_Right:
                if( ctrl ) { if( !repeat ) Command( EditorCommand_RotateRight ); } else { Nudge( 1, 0, 0 ); }
                return true;
            case Qt::Key_Up:
                if( ctrl ) { if( !repeat ) RotateSelection90( true, true ); } else { Nudge( 0, 1, 0 ); }
                return true;
            case Qt::Key_Down:
                if( ctrl ) { if( !repeat ) RotateSelection90( false, true ); } else { Nudge( 0, -1, 0 ); }
                return true;
            case Qt::Key_PageUp:
                Nudge( 0, 0, 1 );
                return true;
            case Qt::Key_PageDown:
                Nudge( 0, 0, -1 );
                return true;
            default:
                break;
        }

        if( repeat ) {
            return false;
        }

        if( ctrl ) {
            switch( key ) {
                case Qt::Key_Z:     Command( shift ? EditorCommand_Redo : EditorCommand_Undo ); return true;
                case Qt::Key_Y:     Command( EditorCommand_Redo ); return true;
                case Qt::Key_S:     Command( shift ? EditorCommand_SaveAs : EditorCommand_Save ); return true;
                case Qt::Key_O:     Command( EditorCommand_Open ); return true;
                case Qt::Key_N:     Command( EditorCommand_New ); return true;
                case Qt::Key_C:     Command( EditorCommand_Copy ); return true;
                case Qt::Key_X:     Command( EditorCommand_Cut ); return true;
                case Qt::Key_V:     Command( EditorCommand_Paste ); return true;
                case Qt::Key_D:     Command( EditorCommand_Duplicate ); return true;
                case Qt::Key_A:     Command( EditorCommand_SelectAll ); return true;
                case Qt::Key_J:     Command( EditorCommand_Merge ); return true;
                case Qt::Key_K:     Command( shift ? EditorCommand_Hollow : EditorCommand_Subtract ); return true;
                case Qt::Key_L:     Command( EditorCommand_Intersect ); return true;
                case Qt::Key_F:     Command( alt ? EditorCommand_FlipVertical : EditorCommand_FlipHorizontal ); return true;
                case Qt::Key_Space: Command( EditorCommand_MaximizePane ); return true;
                default:            return false;
            }
        }

        const i32 gridKey = key - Qt::Key_1;
        if( gridKey >= 0 && gridKey < kGridStepCount && !alt ) {
            SetGridStepIndex( gridKey );
            return true;
        }

        switch( key ) {
            case Qt::Key_Escape:
                if( leftDrag != LeftDrag_None ) {
                    CancelDrag();
                } else if( tool == EditorTool_Clip && clipPointCount > 0 ) {
                    clipPointCount = 0;
                } else if( ( tool == EditorTool_Vertex || tool == EditorTool_Edge || tool == EditorTool_Face ) && handleSelection.count > 0 ) {
                    ListClear( handleSelection );
                } else if( tool != EditorTool_Brush ) {
                    SetTool( EditorTool_Brush );
                } else {
                    Command( EditorCommand_SelectNone );
                }
                return true;
            case Qt::Key_Delete:
            case Qt::Key_Backspace:   Command( EditorCommand_Delete ); return true;
            case Qt::Key_Return:
            case Qt::Key_Enter:       Command( EditorCommand_ClipApply ); return true;
            case Qt::Key_Tab:         Command( EditorCommand_ClipToggleSide ); return true;
            case Qt::Key_C:           Command( EditorCommand_ToolClip ); return true;
            case Qt::Key_V:           Command( EditorCommand_ToolVertex ); return true;
            case Qt::Key_E:           Command( EditorCommand_ToolEdge ); return true;
            case Qt::Key_F:           Command( EditorCommand_ToolFace ); return true;
            case Qt::Key_R:           Command( EditorCommand_ToolRotate ); return true;
            case Qt::Key_B:           Command( EditorCommand_ToolBrush ); return true;
            // Shift rather than Alt: Alt+H is the Help menu's mnemonic.
            case Qt::Key_H:           Command( shift ? EditorCommand_ShowAll : EditorCommand_Hide ); return true;
            case Qt::Key_Z:           Command( EditorCommand_FrameSelection ); return true;
            case Qt::Key_T:           Command( EditorCommand_ToggleTextureLock ); return true;
            case Qt::Key_BracketLeft: Command( EditorCommand_GridFiner ); return true;
            case Qt::Key_BracketRight:Command( EditorCommand_GridCoarser ); return true;
            case Qt::Key_F1:          Command( EditorCommand_LayoutSingle ); return true;
            case Qt::Key_F2:          Command( EditorCommand_LayoutSplit ); return true;
            case Qt::Key_F3:          Command( EditorCommand_LayoutQuad ); return true;
            case Qt::Key_F4:          Command( EditorCommand_LayoutTall ); return true;
            default:                  return false;
        }
    }

    void VulkanView::Command( EditorCommand command ) {
        if( !started && command != EditorCommand_LayoutSingle && command != EditorCommand_LayoutSplit &&
            command != EditorCommand_LayoutQuad && command != EditorCommand_LayoutTall ) {
            return;
        }

        // Anything that edits the map ends a drag first, rather than having
        // the drag's snapshot restored over the top of the edit later.
        CancelDrag();

        switch( command ) {
            case EditorCommand_New:     FileNew(); break;
            case EditorCommand_Open:    FileOpen(); break;
            case EditorCommand_Save:    FileSave(); break;
            case EditorCommand_SaveAs:  FileSaveAs(); break;

            case EditorCommand_Undo:
                if( DocUndo( doc ) ) {
                    PruneHandleSelection();
                    duplicateActive = false;
                    ShowMessage( QStringLiteral( "Undo" ) );
                } else {
                    ShowMessage( QStringLiteral( "Nothing to undo" ) );
                }
                break;
            case EditorCommand_Redo:
                if( DocRedo( doc ) ) {
                    PruneHandleSelection();
                    duplicateActive = false;
                    ShowMessage( QStringLiteral( "Redo" ) );
                } else {
                    ShowMessage( QStringLiteral( "Nothing to redo" ) );
                }
                break;

            case EditorCommand_Copy:
                if( CopySelection() ) {
                    ShowMessage( QStringLiteral( "Copied %1 brush(es)" ).arg( DocSelectedCount( doc ) ) );
                }
                break;
            case EditorCommand_Cut:
                if( CopySelection() && DocDeleteSelected( doc ) ) {
                    ListClear( handleSelection );
                    duplicateActive = false;
                }
                break;
            case EditorCommand_Paste:       PasteClipboard(); break;
            case EditorCommand_Duplicate:   DuplicateSelection(); break;
            case EditorCommand_Delete:
                if( tool == EditorTool_Vertex && handleSelection.count > 0 ) {
                    DeleteSelectedVertices();
                } else if( ( tool == EditorTool_Edge || tool == EditorTool_Face ) && handleSelection.count > 0 ) {
                    // With handles picked the user is working on parts of a
                    // brush, and a Delete that took the whole brush would be
                    // a nasty surprise.
                    ShowMessage( QStringLiteral( "Delete removes corners in the vertex tool; press Escape first to delete brushes" ) );
                } else if( DocDeleteSelected( doc ) ) {
                    ListClear( handleSelection );
                    duplicateActive = false;
                }
                break;
            case EditorCommand_SelectAll:
                DocSelectAll( doc );
                AfterSelectionReplaced();
                break;
            case EditorCommand_SelectNone:
                DocSelectNone( doc );
                AfterSelectionReplaced();
                break;
            case EditorCommand_Hide:
                if( DocHideSelected( doc ) ) {
                    AfterSelectionReplaced();
                }
                break;
            case EditorCommand_ShowAll:
                DocShowAll( doc );
                break;

            case EditorCommand_Merge:
                if( !DocMergeSelected( doc ) ) {
                    ShowMessage( QStringLiteral( "Merge needs two or more brushes selected" ) );
                } else {
                    AfterSelectionReplaced();
                }
                break;
            case EditorCommand_Subtract:
                if( !DocSubtractSelected( doc ) ) {
                    ShowMessage( QStringLiteral( "Subtract: the selection does not overlap anything" ) );
                } else {
                    AfterSelectionReplaced();
                }
                break;
            case EditorCommand_Hollow:
                if( !DocHollowSelected( doc, grid.step ) ) {
                    ShowMessage( QStringLiteral( "Hollow: nothing selected is thicker than two grid steps" ) );
                } else {
                    AfterSelectionReplaced();
                }
                break;
            case EditorCommand_Intersect:
                if( !DocIntersectSelected( doc ) ) {
                    ShowMessage( QStringLiteral( "Intersect needs two or more overlapping brushes" ) );
                } else {
                    AfterSelectionReplaced();
                }
                break;

            case EditorCommand_RotateLeft:      RotateSelection90( true, false ); break;
            case EditorCommand_RotateRight:     RotateSelection90( false, false ); break;
            case EditorCommand_FlipHorizontal:  FlipSelection( false ); break;
            case EditorCommand_FlipVertical:    FlipSelection( true ); break;

            case EditorCommand_ToolBrush:   SetTool( EditorTool_Brush ); break;
            case EditorCommand_ToolClip:    SetTool( EditorTool_Clip ); break;
            case EditorCommand_ToolVertex:  SetTool( EditorTool_Vertex ); break;
            case EditorCommand_ToolEdge:    SetTool( EditorTool_Edge ); break;
            case EditorCommand_ToolFace:    SetTool( EditorTool_Face ); break;
            case EditorCommand_ToolRotate:  SetTool( EditorTool_Rotate ); break;

            case EditorCommand_ClipToggleSide:
                if( tool == EditorTool_Clip ) {
                    clipSide = (ClipSide)( ( clipSide + 1 ) % 3 );
                }
                break;
            case EditorCommand_ClipApply:
                if( tool == EditorTool_Clip ) {
                    ApplyClip();
                }
                break;

            case EditorCommand_GridFiner:   SetGridStepIndex( gridStepIndex - 1 ); break;
            case EditorCommand_GridCoarser: SetGridStepIndex( gridStepIndex + 1 ); break;

            case EditorCommand_FrameSelection: {
                Vec3 min = {};
                Vec3 max = {};
                if( DocSelectionBounds( doc, &min, &max ) || DocVisibleBounds( doc, &min, &max ) ) {
                    FrameBounds( min, max );
                }
                break;
            }

            case EditorCommand_ToggleTextureLock:
                textureLock = !textureLock;
                ShowMessage( textureLock ? QStringLiteral( "Texture lock on" ) : QStringLiteral( "Texture lock off" ) );
                break;

            case EditorCommand_LayoutSingle:    SetLayout( PaneLayout_Single ); break;
            case EditorCommand_LayoutSplit:     SetLayout( PaneLayout_Split ); break;
            case EditorCommand_LayoutQuad:      SetLayout( PaneLayout_Quad ); break;
            case EditorCommand_LayoutTall:      SetLayout( PaneLayout_Tall ); break;
            case EditorCommand_MaximizePane:
                // SetLayout keeps the maximised pane when the layout itself
                // does not change, which is what re-laying the same one does.
                maximizedPane = maximizedPane >= 0 ? -1 : PaneAt( mapFromGlobal( QCursor::pos() ) );
                SetLayout( layout );
                break;

            default:
                break;
        }
    }

    // --- status -------------------------------------------------------------

    static QString FormatNumber( f32 value ) {
        // Grid values are binary fractions; four significant digits show every
        // one of them exactly without printing float noise.
        return QString::number( (double)value, 'g', 5 );
    }

    QString VulkanView::StatusText() const {
        QString text = QStringLiteral( "%1 tool   |   Grid %2   |   Texture lock %3" )
                           .arg( QString::fromUtf8( EditorToolName( tool ) ) )
                           .arg( FormatNumber( grid.step ) )
                           .arg( textureLock ? QStringLiteral( "on" ) : QStringLiteral( "off" ) );

        const i32 selected = DocSelectedCount( doc );
        const i32 faces = DocSelectedFaceCount( doc );
        Vec3 min = {};
        Vec3 max = {};
        if( selected > 0 && DocSelectionBounds( doc, &min, &max ) ) {
            const Vec3 size = max - min;
            text += QStringLiteral( "   |   %1 brush%2   %3 x %4 x %5" )
                        .arg( selected )
                        .arg( selected == 1 ? QString() : QStringLiteral( "es" ) )
                        .arg( FormatNumber( size.x ) )
                        .arg( FormatNumber( size.y ) )
                        .arg( FormatNumber( size.z ) );
        } else if( faces > 0 ) {
            text += QStringLiteral( "   |   %1 face%2" ).arg( faces ).arg( faces == 1 ? QString() : QStringLiteral( "s" ) );
        }

        if( leftDrag == LeftDrag_Create && createValid ) {
            Vec3 cmin = {};
            Vec3 cmax = {};
            BrushBounds( createBrush, &cmin, &cmax );
            const Vec3 size = cmax - cmin;
            text += QStringLiteral( "   |   New brush %1 x %2 x %3%4" )
                        .arg( FormatNumber( size.x ) )
                        .arg( FormatNumber( size.y ) )
                        .arg( FormatNumber( size.z ) )
                        .arg( createHeightMode ? QStringLiteral( "   (height)" ) : QString() );
        } else if( ( leftDrag == LeftDrag_Move || leftDrag == LeftDrag_Handles ) && dragChanged ) {
            text += QStringLiteral( "   |   Moved %1, %2, %3" )
                        .arg( FormatNumber( dragDelta.x ) )
                        .arg( FormatNumber( dragDelta.y ) )
                        .arg( FormatNumber( dragDelta.z ) );
        } else if( leftDrag == LeftDrag_Resize || leftDrag == LeftDrag_Extrude ) {
            text += QStringLiteral( "   |   %1 %2" )
                        .arg( leftDrag == LeftDrag_Resize ? QStringLiteral( "Resize" ) : QStringLiteral( "Extrude" ) )
                        .arg( FormatNumber( faceAmounts[0] ) );
        } else if( leftDrag == LeftDrag_Rotate ) {
            text += QStringLiteral( "   |   Rotate %1 degrees" ).arg( FormatNumber( rotateAngle * kRad2Deg ) );
        } else if( tool == EditorTool_Clip ) {
            const char * sides[3] = { "back", "front", "both" };
            text += QStringLiteral( "   |   %1 point%2, keeping %3" )
                        .arg( clipPointCount )
                        .arg( clipPointCount == 1 ? QString() : QStringLiteral( "s" ) )
                        .arg( QString::fromUtf8( sides[clipSide] ) );
        } else if( tool == EditorTool_Vertex || tool == EditorTool_Edge || tool == EditorTool_Face ) {
            text += QStringLiteral( "   |   %1 handle%2 selected" )
                        .arg( handleSelection.count )
                        .arg( handleSelection.count == 1 ? QString() : QStringLiteral( "s" ) );
        }

        if( !message.isEmpty() && messageTimer.isValid() && messageTimer.elapsed() < 4000 ) {
            text += QStringLiteral( "   |   " ) + message;
        }
        return text;
    }

    QString VulkanView::ToolHint() const {
        switch( tool ) {
            case EditorTool_Brush:
                return QStringLiteral( "Drag: new brush (hold Shift for height)   Drag selection: move (Alt: vertical, Ctrl: copy)   "
                                       "Shift-drag face: resize   Ctrl+Shift-drag face: extrude   Alt+click: paint" );
            case EditorTool_Clip:
                return QStringLiteral( "Click 2 points in a 2D view or 3 on surfaces   Drag points to adjust   Tab: side   Enter: clip" );
            case EditorTool_Vertex:
            case EditorTool_Edge:
            case EditorTool_Face:
                return QStringLiteral( "Click or box-drag handles (Ctrl adds)   Drag handles to reshape (Alt: vertical)   Arrows nudge   Delete removes vertices" );
            case EditorTool_Rotate:
                return QStringLiteral( "Drag a ring to rotate the selection in 15 degree steps (Ctrl: free)" );
            default:
                return QString();
        }
    }

    QString VulkanView::DocumentTitle() const {
        const QString name = filePath.isEmpty() ? QStringLiteral( "untitled" ) : QFileInfo( filePath ).fileName();
        return doc.modified ? name + QStringLiteral( " *" ) : name;
    }

    // --- files --------------------------------------------------------------

    static QString MapsDirectory() {
        const QString directory = QStringLiteral( SOLUM_ASSET_DIR ) + QStringLiteral( "/maps" );
        QDir().mkpath( directory );
        return directory;
    }

    bool VulkanView::ConfirmDiscard() {
        if( !doc.modified ) {
            return true;
        }
        const QMessageBox::StandardButton answer = QMessageBox::question(
            dialogParent, QStringLiteral( "Unsaved changes" ),
            QStringLiteral( "Save changes to %1 first?" ).arg( DocumentTitle() ),
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save );
        requestActivate();
        if( answer == QMessageBox::Cancel ) {
            return false;
        }
        if( answer == QMessageBox::Save ) {
            return FileSave();
        }
        return true;
    }

    bool VulkanView::FileNew() {
        if( !ConfirmDiscard() ) {
            return false;
        }
        Map empty = {};
        DocReplaceMap( doc, empty );
        filePath.clear();
        ListClear( handleSelection );
        clipPointCount = 0;
        duplicateActive = false;
        ShowMessage( QStringLiteral( "New map" ) );
        return true;
    }

    bool VulkanView::FileOpen() {
        if( !ConfirmDiscard() ) {
            return false;
        }
        const QString path = QFileDialog::getOpenFileName( dialogParent, QStringLiteral( "Open Map" ), MapsDirectory(),
                                                           QStringLiteral( "Solum maps (*.smap);;All files (*)" ) );
        requestActivate();
        if( path.isEmpty() ) {
            return false;
        }

        const QByteArray bytes = path.toUtf8();
        Map map = {};
        if( !MapLoad( map, StringView( bytes.constData(), (i32)bytes.size() ) ) ) {
            QMessageBox::warning( dialogParent, QStringLiteral( "Open Map" ), QStringLiteral( "Could not read '%1'." ).arg( path ) );
            return false;
        }
        DocReplaceMap( doc, map );
        filePath = path;
        ListClear( handleSelection );
        clipPointCount = 0;
        duplicateActive = false;

        Vec3 min = {};
        Vec3 max = {};
        if( DocVisibleBounds( doc, &min, &max ) ) {
            FrameBounds( min, max );
        }
        ShowMessage( QStringLiteral( "Opened %1" ).arg( QFileInfo( path ).fileName() ) );
        return true;
    }

    bool VulkanView::FileSave() {
        if( filePath.isEmpty() ) {
            return FileSaveAs();
        }
        const QByteArray bytes = filePath.toUtf8();
        if( !MapSave( doc.map, StringView( bytes.constData(), (i32)bytes.size() ) ) ) {
            QMessageBox::warning( dialogParent, QStringLiteral( "Save Map" ), QStringLiteral( "Could not write '%1'." ).arg( filePath ) );
            return false;
        }
        doc.modified = false;
        ShowMessage( QStringLiteral( "Saved %1" ).arg( QFileInfo( filePath ).fileName() ) );
        return true;
    }

    bool VulkanView::FileSaveAs() {
        QString path = QFileDialog::getSaveFileName( dialogParent, QStringLiteral( "Save Map" ),
                                                     filePath.isEmpty() ? MapsDirectory() : filePath,
                                                     QStringLiteral( "Solum maps (*.smap)" ) );
        requestActivate();
        if( path.isEmpty() ) {
            return false;
        }
        if( !path.endsWith( QStringLiteral( ".smap" ), Qt::CaseInsensitive ) ) {
            path += QStringLiteral( ".smap" );
        }
        filePath = path;
        return FileSave();
    }

    bool VulkanView::CopySelection() {
        if( DocSelectedCount( doc ) == 0 ) {
            return false;
        }
        HeapString text = {};
        MapWriteText( doc.map, true, text );
        lastCopied = QByteArray( text.data, text.count );
        QGuiApplication::clipboard()->setText( QString::fromUtf8( lastCopied ) );
        HeapStringFree( text );
        return true;
    }

    void VulkanView::PasteClipboard() {
        QByteArray bytes = QGuiApplication::clipboard()->text().toUtf8();
        // A clipboard another program holds open, or one the session cannot
        // reach, reads as empty; the last copy made here still pastes. One
        // holding someone else's text is respected and pastes nothing.
        if( bytes.isEmpty() ) {
            bytes = lastCopied;
        }
        if( !bytes.startsWith( "solum_map" ) ) {
            ShowMessage( QStringLiteral( "Nothing to paste: the clipboard holds no brushes" ) );
            return;
        }

        Map pasted = {};
        if( !MapReadText( pasted, StringView( bytes.constData(), (i32)bytes.size() ), "clipboard" ) || pasted.brushes.count == 0 ) {
            MapFree( pasted );
            ShowMessage( QStringLiteral( "The clipboard's brushes could not be read" ) );
            return;
        }

        // In place and selected, so an arrow key or a drag puts them where
        // they are wanted next.
        DocBeginEdit( doc );
        DocSelectNone( doc );
        for( i32 i = 0; i < pasted.brushes.count; i++ ) {
            pasted.brushes[i].flags = BrushFlag_Selected;
            DocAddBrush( doc, pasted.brushes[i] );
        }
        ListFree( pasted.brushes );
        DocEdited( doc );
        AfterSelectionReplaced();
    }

    void VulkanView::CreateStarterMap() {
        // Something to stand on and something to cut into, so a first launch
        // shows what the tools do instead of an empty grid.
        Map map = {};
        FaceTexture floorTexture = FaceTextureDefault();
        Brush floor = {};
        if( BrushCreateBox( floor, Vec3{ -8.0f, -0.5f, -8.0f }, Vec3{ 8.0f, 0.0f, 8.0f }, floorTexture ) ) {
            ListAdd( map.brushes, floor );
        }

        FaceTexture blockTexture = FaceTextureDefault();
        if( QFile::exists( QStringLiteral( SOLUM_ASSET_DIR ) + QStringLiteral( "/T_Bricks1_Color.meta" ) ) ) {
            StringSet( blockTexture.material, "T_Bricks1_Color" );
        }
        Brush block = {};
        if( BrushCreateBox( block, Vec3{ -2.0f, 0.0f, -2.0f }, Vec3{ 2.0f, 3.0f, 2.0f }, blockTexture ) ) {
            ListAdd( map.brushes, block );
        }

        DocReplaceMap( doc, map );
    }

    // --- materials and the inspector -----------------------------------------

    void VulkanView::SetCurrentMaterial( const QString & material, bool applyToSelection ) {
        const QByteArray bytes = material.toUtf8();
        const StringView name( bytes.constData(), (i32)bytes.size() );
        StringSet( currentTexture.material, name );
        if( applyToSelection && started ) {
            CancelDrag();
            DocApplyMaterial( doc, name );
        }
        ShowMessage( QStringLiteral( "Material %1" ).arg( material.isEmpty() ? QStringLiteral( "(none)" ) : material ) );
    }

    QString VulkanView::CurrentMaterial() const {
        return QString::fromUtf8( currentTexture.material.data, currentTexture.material.count );
    }

    bool VulkanView::FaceTextureSummary( FaceTexture * outTexture, i32 * outCount, bool * outMixedMaterial ) const {
        List<FaceRef> faces = {};
        DocTargetFaces( doc, faces );
        const bool any = faces.count > 0;
        if( any ) {
            const FaceTexture & first = doc.map.brushes[faces[0].brush].faces[faces[0].face].texture;
            *outTexture = first;
            *outCount = faces.count;
            *outMixedMaterial = false;
            for( i32 i = 1; i < faces.count; i++ ) {
                if( !StringEquals( doc.map.brushes[faces[i].brush].faces[faces[i].face].texture.material, first.material ) ) {
                    *outMixedMaterial = true;
                    break;
                }
            }
        }
        ListFree( faces );
        return any;
    }

    void VulkanView::SetFaceTextureField( FaceTextureField field, f32 value ) {
        List<FaceRef> faces = {};
        DocTargetFaces( doc, faces );
        if( faces.count == 0 ) {
            ListFree( faces );
            return;
        }

        CancelDrag();
        // A spin box sends a value per tick. Ticks of the same field with
        // nothing in between are one edit, so one undo takes them all back.
        if( lastFieldEdit != (i32)field || lastFieldEditVersion != doc.version ) {
            DocBeginEdit( doc );
        }
        for( i32 i = 0; i < faces.count; i++ ) {
            FaceTexture & texture = doc.map.brushes[faces[i].brush].faces[faces[i].face].texture;
            switch( field ) {
                case FaceTextureField_OffsetU:  texture.offsetU = value; break;
                case FaceTextureField_OffsetV:  texture.offsetV = value; break;
                case FaceTextureField_ScaleU:   texture.scaleU = value; break;
                case FaceTextureField_ScaleV:   texture.scaleV = value; break;
                case FaceTextureField_Rotation: texture.rotation = value; break;
            }
        }
        ListFree( faces );
        DocEdited( doc );
        lastFieldEdit = (i32)field;
        lastFieldEditVersion = doc.version;
    }

    void VulkanView::ResetFaceTextures() {
        List<FaceRef> faces = {};
        DocTargetFaces( doc, faces );
        if( faces.count > 0 ) {
            CancelDrag();
            DocBeginEdit( doc );
            for( i32 i = 0; i < faces.count; i++ ) {
                FaceTexture & texture = doc.map.brushes[faces[i].brush].faces[faces[i].face].texture;
                const SmallString material = texture.material;
                texture = FaceTextureDefault();
                texture.material = material;
            }
            DocEdited( doc );
            lastFieldEdit = -1;
        }
        ListFree( faces );
    }

    void VulkanView::FitFaceTextures() {
        List<FaceRef> faces = {};
        DocTargetFaces( doc, faces );
        if( faces.count > 0 ) {
            CancelDrag();
            DocBeginEdit( doc );
            for( i32 i = 0; i < faces.count; i++ ) {
                const Brush & brush = doc.map.brushes[faces[i].brush];
                BrushFace & face = doc.map.brushes[faces[i].brush].faces[faces[i].face];

                // Measured in unscaled texture axes, then stretched so one
                // repeat covers the face exactly once from its corner.
                FaceTexture unit = face.texture;
                unit.scaleU = 1.0f;
                unit.scaleV = 1.0f;
                Vec3 u = {};
                Vec3 v = {};
                FaceTextureAxes( unit, face.plane.normal, &u, &v );

                f32 minU = 3.4e38f;
                f32 maxU = -3.4e38f;
                f32 minV = 3.4e38f;
                f32 maxV = -3.4e38f;
                for( i32 p = 0; p < face.pointCount; p++ ) {
                    const Vec3 point = brush.points[face.firstPoint + p];
                    minU = Min( minU, Vec3Dot( point, u ) );
                    maxU = Max( maxU, Vec3Dot( point, u ) );
                    minV = Min( minV, Vec3Dot( point, v ) );
                    maxV = Max( maxV, Vec3Dot( point, v ) );
                }
                if( maxU - minU > kBrushEpsilon && maxV - minV > kBrushEpsilon ) {
                    face.texture.scaleU = maxU - minU;
                    face.texture.scaleV = maxV - minV;
                    face.texture.offsetU = -minU / face.texture.scaleU;
                    face.texture.offsetV = -minV / face.texture.scaleV;
                }
            }
            DocEdited( doc );
            lastFieldEdit = -1;
        }
        ListFree( faces );
    }

    // --- Qt events ----------------------------------------------------------

    void VulkanView::keyPressEvent( QKeyEvent * event ) {
        // Auto-repeat would otherwise deliver a release/press pair per repeat,
        // which reads as the key stuttering rather than being held.
        if( !event->isAutoRepeat() ) {
            SetMovementKey( event->key(), true );
        }

        // Modifiers change what a drag does and what hovering shows, without
        // the mouse having moved.
        if( started && !event->isAutoRepeat() &&
            ( event->key() == Qt::Key_Shift || event->key() == Qt::Key_Alt || event->key() == Qt::Key_Control ) ) {
            Qt::KeyboardModifiers modifiers = event->modifiers();
            if( event->key() == Qt::Key_Shift )   { modifiers.setFlag( Qt::ShiftModifier, true ); }
            if( event->key() == Qt::Key_Alt )     { modifiers.setFlag( Qt::AltModifier, true ); }
            if( event->key() == Qt::Key_Control ) { modifiers.setFlag( Qt::ControlModifier, true ); }
            ModifiersChanged( modifiers );
        }

        const bool handled = started && HandleKey( event );

        // Tab is the clip tool's key here, but it is also Qt's focus-navigation
        // key. Handing it to the base class moves focus to the next widget and
        // takes every later key with it, so it always stops here.
        if( handled || event->key() == Qt::Key_Tab || event->key() == Qt::Key_Backtab ) {
            event->accept();
            return;
        }
        QWindow::keyPressEvent( event );
    }

    void VulkanView::keyReleaseEvent( QKeyEvent * event ) {
        if( !event->isAutoRepeat() ) {
            SetMovementKey( event->key(), false );
        }

        if( started && !event->isAutoRepeat() && ( event->key() == Qt::Key_Alt || event->key() == Qt::Key_Control ) ) {
            Qt::KeyboardModifiers modifiers = event->modifiers();
            if( event->key() == Qt::Key_Alt )     { modifiers.setFlag( Qt::AltModifier, false ); }
            if( event->key() == Qt::Key_Control ) { modifiers.setFlag( Qt::ControlModifier, false ); }
            ModifiersChanged( modifiers );
        }

        // Height mode follows Shift while a new brush is being drawn, so its
        // release has to reach the drag as well as the camera.
        if( event->key() == Qt::Key_Shift && started ) {
            Qt::KeyboardModifiers modifiers = event->modifiers();
            modifiers.setFlag( Qt::ShiftModifier, false );
            ModifiersChanged( modifiers );
        }

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
        lastMouse = position;

        // One thing at a time: a camera drag and a tool drag never overlap.
        if( !started || cameraDrag != CameraDrag_None ) {
            QWindow::mousePressEvent( event );
            return;
        }

        activePane = pane;
        const Qt::KeyboardModifiers modifiers = event->modifiers();

        if( event->button() == Qt::RightButton && leftDrag == LeftDrag_None ) {
            CameraDrag kind = CameraDrag_Pan;
            if( !PaneIs2D( pane ) ) {
                kind = ( modifiers & Qt::AltModifier ) ? CameraDrag_Orbit : CameraDrag_Look;
            }
            BeginCameraDrag( kind, pane, position );
            cameraButton = Qt::RightButton;
        } else if( event->button() == Qt::MiddleButton && leftDrag == LeftDrag_None ) {
            BeginCameraDrag( CameraDrag_Pan, pane, position );
            cameraButton = Qt::MiddleButton;
        } else if( event->button() == Qt::LeftButton && leftDrag == LeftDrag_None ) {
            LeftPress( position, pane, modifiers );
        }
        QWindow::mousePressEvent( event );
    }

    void VulkanView::mouseReleaseEvent( QMouseEvent * event ) {
        const QPoint position = event->position().toPoint();
        if( cameraDrag != CameraDrag_None && event->button() == cameraButton ) {
            EndCameraDrag();
        } else if( event->button() == Qt::LeftButton && leftDrag != LeftDrag_None ) {
            LeftRelease( position, event->modifiers() );
        }
        QWindow::mouseReleaseEvent( event );
    }

    void VulkanView::mouseMoveEvent( QMouseEvent * event ) {
        const QPoint position = event->position().toPoint();

        if( cameraDrag != CameraDrag_None ) {
            const QPoint global = event->globalPosition().toPoint();
            const QPoint delta = global - cameraAnchor;
            // The warp below generates its own move event landing exactly on
            // the anchor; ignoring a zero delta is what stops it recursing.
            if( !delta.isNull() ) {
                EditorPane & pane = panes[cameraPane];
                switch( cameraDrag ) {
                    case CameraDrag_Look:
                        pane.flyInput.lookDeltaX += (f32)delta.x();
                        pane.flyInput.lookDeltaY += (f32)delta.y();
                        break;
                    case CameraDrag_Orbit:
                        OrbitBy( (f32)delta.x(), (f32)delta.y() );
                        break;
                    case CameraDrag_Pan:
                        if( PaneIs2D( cameraPane ) ) {
                            pane.orthoInput.panDeltaX += (f32)delta.x();
                            pane.orthoInput.panDeltaY += (f32)delta.y();
                        } else {
                            PanPerspectiveBy( (f32)delta.x(), (f32)delta.y() );
                        }
                        break;
                    default:
                        break;
                }
                QCursor::setPos( cameraAnchor );
            }
            QWindow::mouseMoveEvent( event );
            return;
        }

        if( leftDrag == LeftDrag_None ) {
            activePane = PaneAt( position );
        }

        if( started ) {
            if( leftDrag != LeftDrag_None ) {
                LeftMove( position, event->modifiers() );
            } else {
                HoverMove( position, event->modifiers() );
            }
        }
        lastMouse = position;
        QWindow::mouseMoveEvent( event );
    }

    void VulkanView::wheelEvent( QWheelEvent * event ) {
        const QPoint position = event->position().toPoint();
        const i32 pane = PaneAt( position );
        // A notch is 120 eighths of a degree by Qt's convention.
        const f32 notches = (f32)event->angleDelta().y() / 120.0f;
        if( started && notches != 0.0f ) {
            if( PaneIs2D( pane ) ) {
                ZoomOrthoAt( position, pane, notches );
            } else if( cameraDrag == CameraDrag_Look ) {
                // Mid-flight the wheel sets the speed instead, so the same
                // hand that steers can slow down for detail work.
                FlyCamera & camera = panes[pane].fly;
                camera.moveSpeed *= powf( 1.25f, notches );
                if( camera.moveSpeed < 0.5f )   { camera.moveSpeed = 0.5f; }
                if( camera.moveSpeed > 200.0f ) { camera.moveSpeed = 200.0f; }
                ShowMessage( QStringLiteral( "Fly speed %1" ).arg( FormatNumber( camera.moveSpeed ) ) );
            } else {
                DollyAt( position, pane, notches );
            }
        }
        QWindow::wheelEvent( event );
    }

    void VulkanView::focusOutEvent( QFocusEvent * event ) {
        // Releases arrive at whoever has focus, so a key or button still down
        // when focus leaves would otherwise stick on forever.
        EndCameraDrag();
        CancelDrag();
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

        // QWindowContainer destroys the platform window - and Qt's Vulkan
        // surface with it - before it deletes this window, so the destructor
        // is too late to drop the swapchain built on that surface. This event
        // is the last moment the surface is still alive.
        if( event->type() == QEvent::PlatformSurface &&
            static_cast<QPlatformSurfaceEvent *>( event )->surfaceEventType() == QPlatformSurfaceEvent::SurfaceAboutToBeDestroyed ) {
            if( started ) {
                RendererShutdownDevice( renderer );
                started = false;
                // Nothing brings the device back, so no later frame may try.
                startFailed = true;
            }
        }
        return QWindow::event( event );
    }

} // namespace sol
