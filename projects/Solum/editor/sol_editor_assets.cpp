#include "sol_editor_assets.h"

#include <QDir>
#include <QFileInfo>
#include <QFileSystemModel>
#include <QHeaderView>
#include <QTreeView>
#include <QVBoxLayout>

namespace sol {

    namespace {

        // Shows an asset by name alone. Directories keep their names as they
        // are, since only files carry the .meta extension being hidden.
        class AssetFileModel : public QFileSystemModel {
        public:
            using QFileSystemModel::QFileSystemModel;

            QVariant data( const QModelIndex & index, int role ) const override {
                if( role == Qt::DisplayRole && index.column() == 0 && !isDir( index ) ) {
                    return fileInfo( index ).completeBaseName();
                }
                return QFileSystemModel::data( index, role );
            }
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

} // namespace sol
