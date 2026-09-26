// sol_editor_inspector.h : the texture settings of whatever faces are
// selected, docked beside the viewport.
#pragma once

#include "sol_defines.h"

#include <QWidget>

class QDoubleSpinBox;
class QLabel;

namespace sol {

    class VulkanView;

    // Reads the selected faces - or every face of the selected brushes - and
    // writes edits back through the view, so each one is an undo step like
    // any other edit.
    class FaceInspector : public QWidget {
    public:
        explicit FaceInspector( VulkanView * view, QWidget * parent = nullptr );

        // Re-reads the selection if the document has moved on since the last
        // look. Cheap when it has not, so it can run on a timer.
        void Refresh();

    private:
        VulkanView *        view;
        QLabel *            targetLabel;
        QLabel *            materialLabel;
        QLabel *            currentLabel;
        // Offset U/V, scale U/V, rotation, in FaceTextureField order.
        QDoubleSpinBox *    spins[5];
        u32                 shownVersion;
        QString             shownCurrent;
    };

} // namespace sol
