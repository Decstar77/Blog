// sol_editor_main.cpp : Qt shell hosting the engine's Vulkan viewport.
//

#include "sol_editor_import.h"
#include "sol_editor_view.h"
#include "sol_render.h"

#include <QApplication>
#include <QLabel>
#include <QMainWindow>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QVulkanInstance>
#include <QWidget>

#include <cstdio>
#include <cstring>

namespace {

    // Qt creates the surface, so the instance has to carry the surface
    // extensions Qt's Windows platform plugin will ask for. Spelled out rather
    // than pulled from vulkan_win32.h, which would drag in windows.h.
    const char * const kSurfaceExtensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        "VK_KHR_win32_surface",
    };

    // "editor.exe --import <sourcePath> <outputDirectory> <assetName>". A pure
    // CLI path: it must run headlessly, so it is handled before QApplication
    // (and the Vulkan instance) ever gets constructed.
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

        QLabel * caption = new QLabel(
            QStringLiteral( "Viewport below is the engine's Vulkan renderer, "
                            "clearing to green inside a Qt widget." ) );
        caption->setMargin( 8 );

        QWidget * central = new QWidget;
        QVBoxLayout * layout = new QVBoxLayout( central );
        layout->setContentsMargins( 0, 0, 0, 0 );
        layout->setSpacing( 0 );
        layout->addWidget( caption );
        layout->addWidget( viewport, 1 );

        mainWindow.setCentralWidget( central );
        mainWindow.statusBar()->showMessage( QStringLiteral( "Vulkan viewport" ) );
        mainWindow.resize( 1280, 800 );
        mainWindow.show();

        exitCode = app.exec();
    }
    // The window, and with it Qt's surface, is gone by here; only the instance
    // is left to release.

    vulkanInstance.destroy();
    sol::RendererShutdown( &renderer );
    return exitCode;
}
