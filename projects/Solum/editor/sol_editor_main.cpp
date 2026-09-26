// sol_editor_main.cpp : Qt shell hosting the engine's Vulkan viewport.
//

#include "sol_editor_assets.h"
#include "sol_editor_import.h"
#include "sol_editor_inspector.h"
#include "sol_editor_view.h"
#include "sol_render.h"

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QCloseEvent>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QStatusBar>
#include <QTimer>
#include <QTreeView>
#include <QVulkanInstance>
#include <QWidget>

#include <cstdio>
#include <cstring>

namespace {

    const char * const kSurfaceExtensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        "VK_KHR_win32_surface",
    };

    // Every binding the viewport answers to. The viewport handles its keys
    // itself - it is a separate native window, where Qt's shortcut routing does
    // not reach - so this sheet and HandleKey in sol_editor_view.cpp are the
    // two places a binding lives. Keep them in step.
    const char * const kControlsHelp =
        "<h3>Cameras</h3>"
        "<table>"
        "<tr><td><b>Right-drag</b> (3D)</td><td>Look; hold it and use W A S D, Q/E down/up, Shift faster, wheel sets speed</td></tr>"
        "<tr><td><b>Alt + right-drag</b> (3D)</td><td>Orbit round the point under the cursor</td></tr>"
        "<tr><td><b>Middle-drag</b></td><td>Pan (right-drag pans the 2D views too)</td></tr>"
        "<tr><td><b>Wheel</b></td><td>Dolly towards the cursor (3D), zoom at the cursor (2D)</td></tr>"
        "<tr><td><b>Z</b></td><td>Frame the selection, or everything</td></tr>"
        "<tr><td><b>F1 F2 F3 F4</b></td><td>One pane, two, four, tall 3D with 2D column</td></tr>"
        "<tr><td><b>Ctrl+Space</b></td><td>Maximise the pane under the cursor, and back</td></tr>"
        "</table>"
        "<h3>Brush tool (B, Escape)</h3>"
        "<table>"
        "<tr><td><b>Click</b></td><td>Select; Ctrl+click adds and removes; empty space clears</td></tr>"
        "<tr><td><b>Drag on empty space or a surface</b></td><td>Draw a new box. In 3D it rises off the surface (or the grid); hold Shift to set its height. In 2D its depth comes from the last brush made or selected.</td></tr>"
        "<tr><td><b>Drag the selection</b></td><td>Move it on the grid; Alt drags vertically in 3D; Ctrl drags a copy</td></tr>"
        "<tr><td><b>Shift-drag a face</b></td><td>Resize: move the face along its normal (every selected brush's matching face moves)</td></tr>"
        "<tr><td><b>Drag a selected edge</b> (2D)</td><td>Resize without Shift; near a corner both sides move</td></tr>"
        "<tr><td><b>Ctrl+Shift-drag a face</b></td><td>Extrude a new brush out of it</td></tr>"
        "<tr><td><b>Alt-drag</b></td><td>Box-select brushes wholly inside (Ctrl+Alt adds)</td></tr>"
        "<tr><td><b>Shift+click</b></td><td>Select a face for texturing and pick up its material</td></tr>"
        "<tr><td><b>Alt+click</b></td><td>Paint the current material on a face; Ctrl+Alt+click the whole brush</td></tr>"
        "</table>"
        "<h3>Other tools</h3>"
        "<table>"
        "<tr><td><b>C</b> Clip</td><td>Click two points in a 2D view (or three on surfaces in 3D); drag them to adjust; Tab picks the side kept; Enter cuts</td></tr>"
        "<tr><td><b>V E F</b> Vertex, Edge, Face</td><td>Click or box-drag handles (Ctrl toggles), drag them to reshape (Alt: vertical in 3D), arrows nudge, Delete removes vertices</td></tr>"
        "<tr><td><b>R</b> Rotate</td><td>Drag a ring; 15 degree steps, Ctrl for free rotation</td></tr>"
        "</table>"
        "<h3>Edits</h3>"
        "<table>"
        "<tr><td><b>Arrows, PgUp/PgDn</b></td><td>Nudge one grid step in the pane's own axes</td></tr>"
        "<tr><td><b>Ctrl+D</b></td><td>Duplicate beside the original; move the copy and Ctrl+D again repeats that spacing</td></tr>"
        "<tr><td><b>Ctrl+C, Ctrl+X, Ctrl+V</b></td><td>Copy, cut, paste in place (works between maps and editors)</td></tr>"
        "<tr><td><b>Delete</b></td><td>Delete the selection</td></tr>"
        "<tr><td><b>Ctrl+Left/Right, Ctrl+Up/Down</b></td><td>Rotate 90 degrees about the view axis, or about the view's horizontal</td></tr>"
        "<tr><td><b>Ctrl+F, Ctrl+Alt+F</b></td><td>Flip horizontally, vertically</td></tr>"
        "<tr><td><b>Ctrl+J</b></td><td>Merge the selection into its convex hull</td></tr>"
        "<tr><td><b>Ctrl+K</b></td><td>Subtract the selection from everything it overlaps</td></tr>"
        "<tr><td><b>Ctrl+Shift+K</b></td><td>Hollow the selection into walls one grid step thick</td></tr>"
        "<tr><td><b>Ctrl+L</b></td><td>Intersect the selection</td></tr>"
        "<tr><td><b>H, Shift+H</b></td><td>Hide the selection, show everything</td></tr>"
        "<tr><td><b>1-8, [ ]</b></td><td>Grid size 1/16 to 8</td></tr>"
        "<tr><td><b>T</b></td><td>Texture lock: textures ride along with moved brushes</td></tr>"
        "<tr><td><b>Ctrl+Z, Ctrl+Y</b></td><td>Undo, redo</td></tr>"
        "<tr><td><b>Ctrl+N O S, Ctrl+Shift+S</b></td><td>New, open, save, save as</td></tr>"
        "</table>"
        "<p>Clicking a material in the Assets panel applies it to the selection and makes it the one new brushes wear.</p>";

    // "editor.exe --import <sourcePath> <outputDirectory> <assetName>".
    int RunImportCommand( int argc, char ** argv ) {
        if( argc != 5 ) {
            fprintf( stderr, "usage: editor --import <sourcePath> <outputDirectory> <assetName>\n" );
            return 1;
        }

        const sol::StringView sourcePath = argv[2];
        const sol::StringView outputDirectory = argv[3];
        const sol::StringView assetName = argv[4];

        const bool ok = sol::ImportTexture( sourcePath, outputDirectory, assetName,
                                            sol::TextureFormat_RGBA8_SRGB,
                                            sol::TextureFilter_Linear,
                                            sol::TextureWrap_Repeat );
        if( ok == false ) {
            fprintf( stderr, "Import failed for '%.*s'\n", assetName.count, assetName.data );
            return 1;
        }

        sol::TextureAsset asset = {};
        sol::FixedString<1024> metaPath = {};
        sol::StringAppend( metaPath, outputDirectory );
        sol::StringAppend( metaPath, "/" );
        sol::StringAppend( metaPath, assetName );
        sol::StringAppend( metaPath, ".meta" );

        if( sol::TextureAssetLoad( metaPath, &asset ) ) {
            printf( "Imported '%.*s' -> %.*s\n", sourcePath.count, sourcePath.data, metaPath.count, metaPath.data );
            printf( "  dimensions: %d x %d\n", asset.width, asset.height );
            printf( "  payload bytes: %d\n", asset.pixels.count );
            sol::TextureAssetFree( &asset );
        } else {
            fprintf( stderr, "Import wrote files but round-trip load failed for '%.*s'\n", metaPath.count, metaPath.data );
            return 1;
        }

        return 0;
    }

    // File > Import Texture. Asks for source art and writes the asset pair into
    // the asset directory, named after the source file. The browser picks the
    // new .meta up from the folder on its own.
    void ImportTextureInteractive( QWidget * parent, const QString & assetDirectory ) {
        const QString sourcePath = QFileDialog::getOpenFileName(
            parent, QStringLiteral( "Import Texture" ), QString(),
            QStringLiteral( "Images (*.png *.jpg *.jpeg *.tga *.bmp *.psd *.hdr *.gif);;All files (*)" ) );
        if( sourcePath.isEmpty() ) {
            return;
        }

        // The byte arrays own the UTF-8 the views below borrow, so they have to
        // outlive the import call.
        const QByteArray sourceBytes = sourcePath.toUtf8();
        const QByteArray directoryBytes = assetDirectory.toUtf8();
        const QByteArray nameBytes = QFileInfo( sourcePath ).completeBaseName().toUtf8();

        const bool ok = sol::ImportTexture( sol::StringView( sourceBytes.constData(), (sol::i32)sourceBytes.size() ),
                                            sol::StringView( directoryBytes.constData(), (sol::i32)directoryBytes.size() ),
                                            sol::StringView( nameBytes.constData(), (sol::i32)nameBytes.size() ),
                                            sol::TextureFormat_RGBA8_SRGB,
                                            sol::TextureFilter_Linear,
                                            sol::TextureWrap_Repeat );
        if( !ok ) {
            QMessageBox::warning( parent, QStringLiteral( "Import Texture" ),
                                  QStringLiteral( "Failed to import '%1'." ).arg( sourcePath ) );
        }
    }

    // Asks about unsaved work before the window goes, which a plain
    // QMainWindow has no hook for.
    class EditorMainWindow : public QMainWindow {
    public:
        sol::VulkanView * view = nullptr;

    protected:
        void closeEvent( QCloseEvent * event ) override {
            if( view != nullptr && !view->ConfirmDiscard() ) {
                event->ignore();
                return;
            }
            event->accept();
        }
    };

    struct MenuEntry {
        sol::EditorCommand  command;    // EditorCommand_Count for a separator
        const char *        text;
        const char *        shortcut;
    };

    // Builds one menu from a table. The shortcuts are shown as hints only and
    // scoped to a widget that never has focus: the viewport handles its keys
    // itself, and a live shortcut here would make every binding ambiguous.
    QMenu * AddCommandMenu( QMainWindow & window, sol::VulkanView * view, const char * title,
                            const MenuEntry * entries, size_t entryCount, QAction ** outActions ) {
        QMenu * menu = window.menuBar()->addMenu( QString::fromUtf8( title ) );
        for( size_t i = 0; i < entryCount; i++ ) {
            const MenuEntry & entry = entries[i];
            if( entry.command == sol::EditorCommand_Count ) {
                menu->addSeparator();
                continue;
            }
            QAction * action = menu->addAction( QString::fromUtf8( entry.text ) );
            if( entry.shortcut != nullptr ) {
                action->setShortcut( QKeySequence( QString::fromUtf8( entry.shortcut ) ) );
                action->setShortcutContext( Qt::WidgetShortcut );
            }
            const sol::EditorCommand command = entry.command;
            QObject::connect( action, &QAction::triggered, view, [view, command]() {
                view->Command( command );
                // The menu took focus on the way in, and the viewport is where
                // the keyboard belongs.
                view->requestActivate();
            } );
            if( outActions != nullptr ) {
                outActions[command] = action;
            }
        }
        return menu;
    }

} // namespace

int main( int argc, char ** argv ) {
    if( argc >= 2 && strcmp( argv[1], "--import" ) == 0 ) {
        // No QApplication, no Vulkan instance - importing is headless.
        return RunImportCommand( argc, argv );
    }

    QApplication app( argc, argv );

    sol::Renderer renderer = {};
    if( !sol::RendererCreateInstance( &renderer, kSurfaceExtensions,
                                      (sol::u32)SPLATS_ARRAY_COUNT( kSurfaceExtensions ) ) ) {
        fprintf( stderr, "Failed to create the Vulkan instance\n" );
        return 1;
    }

    // Hand Qt the instance the engine already made, so both sides draw against
    // the same VkInstance rather than Qt standing up a second one.
    QVulkanInstance vulkanInstance;
    vulkanInstance.setVkInstance( renderer.instance );
    if( !vulkanInstance.create() ) {
        fprintf( stderr, "QVulkanInstance::create failed: %d\n", vulkanInstance.errorCode() );
        sol::RendererShutdown( &renderer );
        return 1;
    }

    int exitCode = 0;
    {
        EditorMainWindow mainWindow;
        mainWindow.setWindowTitle( QStringLiteral( "Solum Editor" ) );

        sol::VulkanView * view = new sol::VulkanView( &renderer );
        view->setVulkanInstance( &vulkanInstance );
        view->SetDialogParent( &mainWindow );
        mainWindow.view = view;

        QWidget * viewport = QWidget::createWindowContainer( view );
        viewport->setMinimumSize( 640, 360 );
        viewport->setFocusPolicy( Qt::StrongFocus );
        mainWindow.setCentralWidget( viewport );

        const QString assetDirectory = QStringLiteral( SOLUM_ASSET_DIR );
        sol::AssetBrowser * assetBrowser = new sol::AssetBrowser( assetDirectory );

        QDockWidget * assetDock = new QDockWidget( QStringLiteral( "Assets" ), &mainWindow );
        assetDock->setObjectName( QStringLiteral( "AssetDock" ) );
        assetDock->setAllowedAreas( Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea );
        assetDock->setWidget( assetBrowser );
        mainWindow.addDockWidget( Qt::LeftDockWidgetArea, assetDock );

        // A click on a material is the whole texturing workflow: it goes onto
        // the selection and becomes what the next brush wears. The keyboard is
        // handed straight back so the next key is a tool key, not a tree search.
        QObject::connect( assetBrowser->treeView(), &QTreeView::clicked, view, [view, assetBrowser]( const QModelIndex & index ) {
            const QString name = assetBrowser->assetName( index );
            if( !name.isEmpty() ) {
                view->SetCurrentMaterial( name, true );
                view->requestActivate();
            }
        } );

        sol::FaceInspector * inspector = new sol::FaceInspector( view );
        QDockWidget * inspectorDock = new QDockWidget( QStringLiteral( "Face" ), &mainWindow );
        inspectorDock->setObjectName( QStringLiteral( "InspectorDock" ) );
        inspectorDock->setAllowedAreas( Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea );
        inspectorDock->setWidget( inspector );
        mainWindow.addDockWidget( Qt::LeftDockWidgetArea, inspectorDock );
        mainWindow.splitDockWidget( assetDock, inspectorDock, Qt::Vertical );
        mainWindow.resizeDocks( { assetDock }, { 260 }, Qt::Horizontal );

        QAction * commandActions[sol::EditorCommand_Count] = {};
        const sol::EditorCommand kSeparator = sol::EditorCommand_Count;

        const MenuEntry fileEntries[] = {
            { sol::EditorCommand_New,    "&New",         "Ctrl+N" },
            { sol::EditorCommand_Open,   "&Open...",     "Ctrl+O" },
            { sol::EditorCommand_Save,   "&Save",        "Ctrl+S" },
            { sol::EditorCommand_SaveAs, "Save &As...",  "Ctrl+Shift+S" },
        };
        QMenu * fileMenu = AddCommandMenu( mainWindow, view, "&File", fileEntries, SPLATS_ARRAY_COUNT( fileEntries ), commandActions );
        fileMenu->addSeparator();
        QAction * importAction = fileMenu->addAction( QStringLiteral( "&Import Texture..." ) );
        QObject::connect( importAction, &QAction::triggered, &mainWindow, [&mainWindow, assetDirectory]() {
            ImportTextureInteractive( &mainWindow, assetDirectory );
        } );
        fileMenu->addSeparator();
        QAction * exitAction = fileMenu->addAction( QStringLiteral( "E&xit" ) );
        QObject::connect( exitAction, &QAction::triggered, &mainWindow, &QMainWindow::close );

        const MenuEntry editEntries[] = {
            { sol::EditorCommand_Undo,       "&Undo",        "Ctrl+Z" },
            { sol::EditorCommand_Redo,       "&Redo",        "Ctrl+Y" },
            { kSeparator,                    nullptr,        nullptr },
            { sol::EditorCommand_Cut,        "Cu&t",         "Ctrl+X" },
            { sol::EditorCommand_Copy,       "&Copy",        "Ctrl+C" },
            { sol::EditorCommand_Paste,      "&Paste",       "Ctrl+V" },
            { sol::EditorCommand_Duplicate,  "D&uplicate",   "Ctrl+D" },
            { sol::EditorCommand_Delete,     "&Delete",      "Del" },
            { kSeparator,                    nullptr,        nullptr },
            { sol::EditorCommand_SelectAll,  "Select &All",  "Ctrl+A" },
            { sol::EditorCommand_SelectNone, "Select &None", "Esc" },
            { kSeparator,                    nullptr,        nullptr },
            { sol::EditorCommand_Hide,       "&Hide",        "H" },
            { sol::EditorCommand_ShowAll,    "Show All",     "Shift+H" },
        };
        AddCommandMenu( mainWindow, view, "&Edit", editEntries, SPLATS_ARRAY_COUNT( editEntries ), commandActions );

        const MenuEntry brushEntries[] = {
            { sol::EditorCommand_Merge,          "&Merge (convex hull)",   "Ctrl+J" },
            { sol::EditorCommand_Subtract,       "&Subtract",              "Ctrl+K" },
            { sol::EditorCommand_Hollow,         "&Hollow",                "Ctrl+Shift+K" },
            { sol::EditorCommand_Intersect,      "&Intersect",             "Ctrl+L" },
            { kSeparator,                        nullptr,                  nullptr },
            { sol::EditorCommand_RotateLeft,     "Rotate 90 Counter-clockwise", "Ctrl+Left" },
            { sol::EditorCommand_RotateRight,    "Rotate 90 Clockwise",    "Ctrl+Right" },
            { sol::EditorCommand_FlipHorizontal, "Flip &Horizontally",     "Ctrl+F" },
            { sol::EditorCommand_FlipVertical,   "Flip &Vertically",       "Ctrl+Alt+F" },
            { kSeparator,                        nullptr,                  nullptr },
            { sol::EditorCommand_ToggleTextureLock, "&Texture Lock",       "T" },
        };
        AddCommandMenu( mainWindow, view, "&Brush", brushEntries, SPLATS_ARRAY_COUNT( brushEntries ), commandActions );
        commandActions[sol::EditorCommand_ToggleTextureLock]->setCheckable( true );

        const MenuEntry toolEntries[] = {
            { sol::EditorCommand_ToolBrush,  "&Brush",   "B" },
            { sol::EditorCommand_ToolClip,   "&Clip",    "C" },
            { sol::EditorCommand_ToolVertex, "&Vertex",  "V" },
            { sol::EditorCommand_ToolEdge,   "&Edge",    "E" },
            { sol::EditorCommand_ToolFace,   "&Face",    "F" },
            { sol::EditorCommand_ToolRotate, "&Rotate",  "R" },
            { kSeparator,                    nullptr,    nullptr },
            { sol::EditorCommand_ClipToggleSide, "Clip: Toggle Kept Side", "Tab" },
            { sol::EditorCommand_ClipApply,      "Clip: Cut",              "Return" },
        };
        AddCommandMenu( mainWindow, view, "&Tools", toolEntries, SPLATS_ARRAY_COUNT( toolEntries ), commandActions );
        QActionGroup * toolGroup = new QActionGroup( &mainWindow );
        toolGroup->setExclusive( true );
        const sol::EditorCommand toolCommands[sol::EditorTool_Count] = {
            sol::EditorCommand_ToolBrush, sol::EditorCommand_ToolClip, sol::EditorCommand_ToolVertex,
            sol::EditorCommand_ToolEdge, sol::EditorCommand_ToolFace, sol::EditorCommand_ToolRotate,
        };
        for( int t = 0; t < sol::EditorTool_Count; t++ ) {
            commandActions[toolCommands[t]]->setCheckable( true );
            toolGroup->addAction( commandActions[toolCommands[t]] );
        }

        const MenuEntry viewEntries[] = {
            { sol::EditorCommand_FrameSelection, "&Frame Selection",   "Z" },
            { sol::EditorCommand_GridFiner,      "Grid &Finer",        "[" },
            { sol::EditorCommand_GridCoarser,    "Grid &Coarser",      "]" },
            { kSeparator,                        nullptr,              nullptr },
            { sol::EditorCommand_LayoutSingle,   "&Single Pane",       "F1" },
            { sol::EditorCommand_LayoutSplit,    "S&plit",             "F2" },
            { sol::EditorCommand_LayoutQuad,     "&Quad",              "F3" },
            { sol::EditorCommand_LayoutTall,     "&Tall 3D + 2D Column", "F4" },
            { sol::EditorCommand_MaximizePane,   "&Maximize Pane",     "Ctrl+Space" },
        };
        QMenu * viewMenu = AddCommandMenu( mainWindow, view, "&View", viewEntries, SPLATS_ARRAY_COUNT( viewEntries ), commandActions );
        QActionGroup * layoutGroup = new QActionGroup( &mainWindow );
        layoutGroup->setExclusive( true );
        const sol::EditorCommand layoutCommands[sol::PaneLayout_Count] = {
            sol::EditorCommand_LayoutSingle, sol::EditorCommand_LayoutSplit, sol::EditorCommand_LayoutQuad, sol::EditorCommand_LayoutTall,
        };
        for( int l = 0; l < sol::PaneLayout_Count; l++ ) {
            commandActions[layoutCommands[l]]->setCheckable( true );
            layoutGroup->addAction( commandActions[layoutCommands[l]] );
        }
        viewMenu->addSeparator();
        // The docks' own toggle actions, so the ticks always match whether the
        // panels are actually showing, including after closing one by its X.
        viewMenu->addAction( assetDock->toggleViewAction() );
        viewMenu->addAction( inspectorDock->toggleViewAction() );

        QMenu * helpMenu = mainWindow.menuBar()->addMenu( QStringLiteral( "&Help" ) );
        QAction * controlsAction = helpMenu->addAction( QStringLiteral( "&Controls" ) );
        QObject::connect( controlsAction, &QAction::triggered, &mainWindow, [&mainWindow]() {
            QMessageBox box( &mainWindow );
            box.setWindowTitle( QStringLiteral( "Controls" ) );
            box.setTextFormat( Qt::RichText );
            box.setText( QString::fromUtf8( kControlsHelp ) );
            box.exec();
        } );

        // The status bar carries what the viewport cannot draw as text: the
        // tool, the grid, what is selected and what a drag is measuring, with
        // the current tool's buttons on the right.
        QLabel * statusLabel = new QLabel( &mainWindow );
        QLabel * hintLabel = new QLabel( &mainWindow );
        hintLabel->setStyleSheet( QStringLiteral( "color: gray;" ) );
        mainWindow.statusBar()->addWidget( statusLabel, 1 );
        mainWindow.statusBar()->addPermanentWidget( hintLabel );

        // Polled rather than signalled: the view is a plain QWindow with no
        // signals of its own, and ten times a second is plenty for text.
        QTimer * refreshTimer = new QTimer( &mainWindow );
        QObject::connect( refreshTimer, &QTimer::timeout, &mainWindow,
                          [&mainWindow, view, statusLabel, hintLabel, inspector, commandActions, toolCommands, layoutCommands]() {
            statusLabel->setText( view->StatusText() );
            hintLabel->setText( view->ToolHint() );
            mainWindow.setWindowTitle( view->DocumentTitle() + QStringLiteral( " - Solum Editor" ) );
            inspector->Refresh();
            commandActions[toolCommands[view->CurrentTool()]]->setChecked( true );
            commandActions[layoutCommands[view->CurrentLayout()]]->setChecked( true );
            commandActions[sol::EditorCommand_ToggleTextureLock]->setChecked( view->TextureLock() );
        } );
        refreshTimer->start( 100 );

        mainWindow.showMaximized();
        exitCode = app.exec();
    }
    // The window, and with it Qt's surface, is gone by here; only the instance
    // is left to release.

    vulkanInstance.destroy();
    sol::RendererShutdown( &renderer );
    return exitCode;
}
