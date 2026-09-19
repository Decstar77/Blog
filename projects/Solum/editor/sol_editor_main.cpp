// sol_editor_main.cpp : Qt shell hosting the engine's Vulkan viewport.
//

#include "sol_editor_assets.h"
#include "sol_editor_import.h"
#include "sol_editor_view.h"
#include "sol_render.h"

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QStatusBar>
#include <QVulkanInstance>
#include <QWidget>

#include <cstdio>
#include <cstring>

namespace {

    const char * const kSurfaceExtensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        "VK_KHR_win32_surface",
    };

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
        QMainWindow mainWindow;
        mainWindow.setWindowTitle( QStringLiteral( "Solum Editor" ) );

        sol::VulkanView * view = new sol::VulkanView( &renderer );
        view->setVulkanInstance( &vulkanInstance );

        QWidget * viewport = QWidget::createWindowContainer( view );
        viewport->setMinimumSize( 640, 360 );
        viewport->setFocusPolicy( Qt::StrongFocus );

        // The viewport is the centre of the window. The controls live in the
        // status bar rather than a caption strip taking a row off the render.
        mainWindow.setCentralWidget( viewport );

        const QString assetDirectory = QStringLiteral( SOLUM_ASSET_DIR );
        sol::AssetBrowser * assetBrowser = new sol::AssetBrowser( assetDirectory );

        QDockWidget * assetDock = new QDockWidget( QStringLiteral( "Assets" ), &mainWindow );
        assetDock->setObjectName( QStringLiteral( "AssetDock" ) );
        assetDock->setAllowedAreas( Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea );
        assetDock->setWidget( assetBrowser );
        mainWindow.addDockWidget( Qt::LeftDockWidgetArea, assetDock );
        mainWindow.resizeDocks( { assetDock }, { 280 }, Qt::Horizontal );

        QMenu * fileMenu = mainWindow.menuBar()->addMenu( QStringLiteral( "&File" ) );
        QAction * importAction = fileMenu->addAction( QStringLiteral( "&Import Texture..." ) );
        QObject::connect( importAction, &QAction::triggered, &mainWindow, [&mainWindow, assetDirectory]() {
            ImportTextureInteractive( &mainWindow, assetDirectory );
        } );
        fileMenu->addSeparator();
        QAction * exitAction = fileMenu->addAction( QStringLiteral( "E&xit" ) );
        QObject::connect( exitAction, &QAction::triggered, &mainWindow, &QMainWindow::close );

        // The dock's own toggle action, so the menu tick always matches whether
        // the panel is actually showing, including after closing it by its X.
        QMenu * viewMenu = mainWindow.menuBar()->addMenu( QStringLiteral( "&View" ) );
        viewMenu->addAction( assetDock->toggleViewAction() );

        // The same three layouts the function keys reach. Exclusive, so the
        // menu reads as a choice rather than three independent toggles.
        QMenu * layoutMenu = viewMenu->addMenu( QStringLiteral( "&Layout" ) );
        QActionGroup * layoutGroup = new QActionGroup( layoutMenu );
        layoutGroup->setExclusive( true );

        struct LayoutEntry {
            sol::PaneLayout layout;
            const char *    text;
            const char *    shortcut;
        };
        const LayoutEntry layoutEntries[] = {
            { sol::PaneLayout_Single, "&Single",          "F1" },
            { sol::PaneLayout_Split,  "S&plit",           "F2" },
            { sol::PaneLayout_Quad,   "&Quad",            "F3" },
        };

        for( size_t i = 0; i < SPLATS_ARRAY_COUNT( layoutEntries ); i++ ) {
            const LayoutEntry & entry = layoutEntries[i];
            QAction * action = layoutMenu->addAction( QString::fromUtf8( entry.text ) );
            action->setCheckable( true );
            action->setChecked( view->CurrentLayout() == entry.layout );
            // The layout each action names, so the refresh below can read it
            // back off the group without a parallel array to keep in step.
            action->setData( (int)entry.layout );
            // A hint only, and deliberately scoped to a widget that never has
            // focus: the viewport handles the function keys itself, and a live
            // shortcut here would make the binding ambiguous.
            action->setShortcut( QKeySequence( QString::fromUtf8( entry.shortcut ) ) );
            action->setShortcutContext( Qt::WidgetShortcut );
            layoutGroup->addAction( action );

            const sol::PaneLayout target = entry.layout;
            QObject::connect( action, &QAction::triggered, view, [view, target]() {
                view->SetLayout( target );
                // The menu took focus on the way in, and the viewport is where
                // the keyboard belongs.
                view->requestActivate();
            } );
        }

        // The function keys change the layout without going through the menu,
        // so the ticks are refreshed on the way in rather than only on click.
        QObject::connect( layoutMenu, &QMenu::aboutToShow, layoutMenu, [view, layoutGroup]() {
            const QList<QAction *> actions = layoutGroup->actions();
            for( QAction * action : actions ) {
                action->setChecked( (int)view->CurrentLayout() == action->data().toInt() );
            }
        } );

        mainWindow.statusBar()->showMessage(
            QStringLiteral( "F1/F2/F3 switch to one, two and four panes. "
                            "The pane under the cursor takes the input.    "
                            "Perspective pane: WASD to move, right-drag to look, "
                            "Space/Ctrl for up and down, Shift to sprint.    "
                            "Top, front and side panes: right-drag to pan, wheel to zoom.    "
                            "Left-click to select, click empty space to deselect.    "
                            "B toggles build mode: drag out a base on the grid, release to lock it, "
                            "move to extrude it into a box, click to finish, Escape to cancel.    "
                            "Delete removes the selection, Tab toggles edit mode, which locks it.    "
                            "Build and edit mode are exclusive: entering one leaves the other.    "
                            "In edit mode, click a vertex to select it and T to move it.    "
                            "T and R put the move and rotate gizmo on the selection, "
                            "rotation snapping to 15 degrees unless Ctrl is held.    "
                            "Keys 1-6 set the grid size, G cycles the grid's plane, "
                            "Alt+click lands the grid on the face under the cursor." ) );
        mainWindow.showMaximized();

        exitCode = app.exec();
    }
    // The window, and with it Qt's surface, is gone by here; only the instance
    // is left to release.

    vulkanInstance.destroy();
    sol::RendererShutdown( &renderer );
    return exitCode;
}
