using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;

namespace Editor.Panels {
    // Captures Console.Out / Console.Error (the engine logs through them) into a read-only text pane.
    public sealed class ConsolePanel : UserControl {
        private readonly TextBox log;

        public ConsolePanel() {
            log = new TextBox {
                IsReadOnly = true,
                FontFamily = new FontFamily( "Cascadia Mono, Consolas" ),
                FontSize = 12,
                BorderThickness = new Thickness( 0 ),
                Background = (Brush)Application.Current.FindResource( "PanelBrush" ),
                Foreground = (Brush)Application.Current.FindResource( "TextBrush" ),
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
                Padding = new Thickness( 6 ),
            };
            Content = log;

            Console.SetOut( new PanelWriter( this, "" ) );
            Console.SetError( new PanelWriter( this, "error: " ) );
        }

        private void Append( string line ) {
            log.AppendText( $"[{DateTime.Now:HH:mm:ss}] {line}{Environment.NewLine}" );
            log.ScrollToEnd();
        }

        private sealed class PanelWriter : TextWriter {
            private readonly ConsolePanel panel;
            private readonly string prefix;
            private readonly StringBuilder pending = new();

            public PanelWriter( ConsolePanel panel, string prefix ) {
                this.panel = panel;
                this.prefix = prefix;
            }

            public override Encoding Encoding => Encoding.UTF8;

            public override void Write( char value ) {
                lock ( pending ) {
                    if ( value == '\n' ) {
                        string line = prefix + pending.ToString().TrimEnd( '\r' );
                        pending.Clear();
                        panel.Dispatcher.BeginInvoke( DispatcherPriority.Background, () => panel.Append( line ) );
                    } else {
                        pending.Append( value );
                    }
                }
            }
        }
    }
}
