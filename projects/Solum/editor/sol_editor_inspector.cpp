#include "sol_editor_inspector.h"
#include "sol_editor_view.h"

#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QVBoxLayout>

namespace sol {

    FaceInspector::FaceInspector( VulkanView * view, QWidget * parent )
        : QWidget( parent ), view( view ), targetLabel( nullptr ), materialLabel( nullptr ), currentLabel( nullptr ),
          spins(), shownVersion( 0 ) {
        QVBoxLayout * layout = new QVBoxLayout( this );

        targetLabel = new QLabel( this );
        materialLabel = new QLabel( this );
        materialLabel->setWordWrap( true );
        currentLabel = new QLabel( this );
        currentLabel->setWordWrap( true );
        layout->addWidget( targetLabel );
        layout->addWidget( materialLabel );

        QFormLayout * form = new QFormLayout();
        const char * const names[5] = { "Offset U", "Offset V", "Scale U", "Scale V", "Rotation" };
        for( i32 i = 0; i < 5; i++ ) {
            QDoubleSpinBox * spin = new QDoubleSpinBox( this );
            spin->setDecimals( 4 );
            if( i < 2 ) {
                spin->setRange( -1000.0, 1000.0 );
                spin->setSingleStep( 0.0625 );
            } else if( i < 4 ) {
                spin->setRange( -1000.0, 1000.0 );
                spin->setSingleStep( 0.125 );
            } else {
                spin->setRange( -360.0, 360.0 );
                spin->setSingleStep( 15.0 );
                spin->setDecimals( 2 );
            }
            // Only settled values - Enter, focus leaving, the arrows - reach the
            // map. Tracking every keystroke would make "1.25" three edits and
            // pass through 1 and 1.2 on the way.
            spin->setKeyboardTracking( false );
            spins[i] = spin;
            form->addRow( QString::fromUtf8( names[i] ), spin );

            const FaceTextureField field = (FaceTextureField)i;
            QObject::connect( spin, &QDoubleSpinBox::valueChanged, this, [this, field]( double value ) {
                // A value of zero would collapse the texture; it is refused
                // here so the map never holds one.
                if( ( field == FaceTextureField_ScaleU || field == FaceTextureField_ScaleV ) && value == 0.0 ) {
                    return;
                }
                this->view->SetFaceTextureField( field, (f32)value );
            } );
        }
        layout->addLayout( form );

        QHBoxLayout * buttons = new QHBoxLayout();
        QPushButton * reset = new QPushButton( QStringLiteral( "Reset" ), this );
        QPushButton * fit = new QPushButton( QStringLiteral( "Fit" ), this );
        reset->setToolTip( QStringLiteral( "Back to one repeat per unit, no offset or rotation" ) );
        fit->setToolTip( QStringLiteral( "Stretch the texture to cover each face exactly once" ) );
        buttons->addWidget( reset );
        buttons->addWidget( fit );
        layout->addLayout( buttons );
        QObject::connect( reset, &QPushButton::clicked, this, [this]() { this->view->ResetFaceTextures(); this->view->requestActivate(); } );
        QObject::connect( fit, &QPushButton::clicked, this, [this]() { this->view->FitFaceTextures(); this->view->requestActivate(); } );

        layout->addSpacing( 8 );
        layout->addWidget( currentLabel );
        layout->addStretch( 1 );

        Refresh();
    }

    void FaceInspector::Refresh() {
        const QString current = view->CurrentMaterial();
        if( current != shownCurrent ) {
            shownCurrent = current;
            currentLabel->setText( QStringLiteral( "New brushes and Alt+click paint: %1" )
                                       .arg( current.isEmpty() ? QStringLiteral( "(dev grid)" ) : current ) );
        }

        if( view->DocVersion() == shownVersion ) {
            return;
        }
        shownVersion = view->DocVersion();

        FaceTexture texture = FaceTextureDefault();
        i32 count = 0;
        bool mixed = false;
        const bool any = view->FaceTextureSummary( &texture, &count, &mixed );

        targetLabel->setText( any ? QStringLiteral( "%1 face%2" ).arg( count ).arg( count == 1 ? QString() : QStringLiteral( "s" ) )
                                  : QStringLiteral( "Select brushes, or Shift+click faces" ) );
        const QString material = QString::fromUtf8( texture.material.data, texture.material.count );
        materialLabel->setText( !any ? QString()
                                     : mixed ? QStringLiteral( "Material: (several)" )
                                             : QStringLiteral( "Material: %1" ).arg( material.isEmpty() ? QStringLiteral( "(dev grid)" ) : material ) );

        const f32 values[5] = { texture.offsetU, texture.offsetV, texture.scaleU, texture.scaleV, texture.rotation };
        for( i32 i = 0; i < 5; i++ ) {
            spins[i]->setEnabled( any );
            // The box being typed into is left alone, so a refresh cannot
            // yank the text out from under the cursor.
            if( spins[i]->hasFocus() ) {
                continue;
            }
            const QSignalBlocker blocker( spins[i] );
            spins[i]->setValue( (double)values[i] );
        }
    }

} // namespace sol
