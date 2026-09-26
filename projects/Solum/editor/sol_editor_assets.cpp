#include "sol_editor_assets.h"
#include "sol_asset.h"

#include <QDir>
#include <QFileInfo>
#include <QFileSystemModel>
#include <QHash>
#include <QHeaderView>
#include <QIcon>
#include <QImage>
#include <QPixmap>
#include <QTreeView>
#include <QVBoxLayout>

namespace sol {

    namespace {

        constexpr int kThumbnailSize = 48;

        // Shows an asset by name alone, with a thumbnail of what it looks
        // like: picking a material by eye is most of what this panel is for.
        // Directories keep their names and icons as they are.
        class AssetFileModel : public QFileSystemModel {
        public:
            using QFileSystemModel::QFileSystemModel;

            QVariant data( const QModelIndex & index, int role ) const override {
                if( index.column() == 0 && !isDir( index ) ) {
                    if( role == Qt::DisplayRole ) {
                        return fileInfo( index ).completeBaseName();
                    }
                    if( role == Qt::DecorationRole ) {
                        return Thumbnail( filePath( index ) );
                    }
                }
                return QFileSystemModel::data( index, role );
            }

        private:
            // Read through the engine's own loader, from the .stex payload the
            // importer wrote, so the thumbnail is exactly what the renderer
            // will sample. Cached by path: a payload is decoded once.
            QIcon Thumbnail( const QString & metaPath ) const {
                const auto found = thumbnails.constFind( metaPath );
                if( found != thumbnails.constEnd() ) {
                    return found.value();
                }

                QIcon icon;
                const QByteArray path = metaPath.toUtf8();
                TextureAsset asset = {};
                if( TextureAssetLoad( StringView( path.constData(), (i32)path.size() ), &asset ) ) {
                    const QImage image( asset.pixels.data, asset.width, asset.height, asset.width * 4, QImage::Format_RGBA8888 );
                    icon = QIcon( QPixmap::fromImage( image.scaled( kThumbnailSize, kThumbnailSize, Qt::KeepAspectRatio, Qt::SmoothTransformation ) ) );
                    TextureAssetFree( &asset );
                }
                thumbnails.insert( metaPath, icon );
                return icon;
            }

            mutable QHash<QString, QIcon> thumbnails;
        };

    } // namespace

    AssetBrowser::AssetBrowser( const QString & rootDirectory, QWidget * parent )
        : QWidget( parent ), root( QDir::cleanPath( rootDirectory ) ), model( nullptr ), tree( nullptr ) {
        // Created up front, so the model below always has somewhere to watch
        // and the importer always has somewhere to write.
        QDir().mkpath( root );

        model = new AssetFileModel( this );
        model->setFilter( QDir::AllDirs | QDir::Files | QDir::NoDotAndDotDot );
        model->setNameFilters( { QStringLiteral( "*.meta" ) } );
        // Hide what does not match instead of greying it out, or every .stex
        // would sit in the list next to the asset it belongs to.
        model->setNameFilterDisables( false );
        model->setRootPath( root );

        tree = new QTreeView( this );
        tree->setModel( model );
        tree->setRootIndex( model->index( root ) );
        tree->setHeaderHidden( true );
        tree->setIconSize( QSize( kThumbnailSize, kThumbnailSize ) );
        tree->setSortingEnabled( true );
        tree->sortByColumn( 0, Qt::AscendingOrder );
        // Size, type and date say nothing useful about an asset.
        for( int column = 1; column < model->columnCount(); ++column ) {
            tree->hideColumn( column );
        }

        QVBoxLayout * layout = new QVBoxLayout( this );
        layout->setContentsMargins( 0, 0, 0, 0 );
        layout->addWidget( tree );
    }

    QString AssetBrowser::assetName( const QModelIndex & index ) const {
        if( !index.isValid() || model->isDir( index ) ) {
            return QString();
        }
        const QString relative = QDir( root ).relativeFilePath( model->filePath( index ) );
        const QFileInfo info( relative );
        const QString directory = info.path();
        // Forward slashes whatever the platform, since the name is stored in
        // map files that have to open anywhere.
        return directory == QStringLiteral( "." ) ? info.completeBaseName()
                                                   : QDir::fromNativeSeparators( directory ) + QStringLiteral( "/" ) + info.completeBaseName();
    }

} // namespace sol
