#pragma once

#include "sol_render.h"

#include <QWindow>

namespace sol {

    // A QWindow with a Vulkan surface type, so Qt creates the native window and
    // the VkSurfaceKHR and the engine renderer draws straight into it. Wrapped
    // by QWidget::createWindowContainer to sit inside ordinary Qt layouts.
    class VulkanView : public QWindow {
    public:
        explicit VulkanView( Renderer * renderer );
        ~VulkanView() override;

        bool startupFailed() const { return startFailed; }

    protected:
        void exposeEvent( QExposeEvent * event ) override;
        void resizeEvent( QResizeEvent * event ) override;
        bool event( QEvent * event ) override;

    private:
        bool EnsureStarted();
        void Render();

        Renderer *  renderer;
        bool        started;
        bool        startFailed;
    };

} // namespace sol
