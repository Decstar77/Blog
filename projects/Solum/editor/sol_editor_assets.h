// sol_editor_assets.h : the asset browser docked down the left of the editor.
#pragma once

#include <QString>
#include <QWidget>

class QFileSystemModel;
class QTreeView;

namespace sol {

    // Lists the engine assets under one directory. It reads the folder itself
    // rather than keeping its own index, so anything the importer writes shows
    // up without being told about it.
    //
    // Only .meta files are listed, named without the extension: the sidecar is
    // the asset as far as the engine is concerned, and the .stex beside it is
    // just its payload.
    class AssetBrowser : public QWidget {
    public:
        explicit AssetBrowser( const QString & rootDirectory, QWidget * parent = nullptr );

        const QString & rootDirectory() const { return root; }

    private:
        QString             root;
        QFileSystemModel *  model;
        QTreeView *         tree;
    };

} // namespace sol
