using System.Windows;
using System.Windows.Controls;

namespace Editor.Core {
    // Minimal single-line text prompt (WPF has no built-in InputBox).
    public static class PromptDialog {
        public static string? Ask( string title, string label, string initial = "" ) {
            TextBox input = new TextBox { Text = initial, Margin = new Thickness( 0, 6, 0, 12 ), MinWidth = 320 };
            Button ok = new Button { Content = "OK", IsDefault = true, MinWidth = 80, Margin = new Thickness( 0, 0, 8, 0 ) };
            Button cancel = new Button { Content = "Cancel", IsCancel = true, MinWidth = 80 };

            Window window = new Window {
                Title = title,
                Owner = Application.Current.MainWindow,
                WindowStartupLocation = WindowStartupLocation.CenterOwner,
                SizeToContent = SizeToContent.WidthAndHeight,
                ResizeMode = ResizeMode.NoResize,
                ShowInTaskbar = false,
                Background = (System.Windows.Media.Brush)Application.Current.FindResource( "PanelBrush" ),
                Content = new StackPanel {
                    Margin = new Thickness( 16 ),
                    Children = {
                        new TextBlock { Text = label },
                        input,
                        new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right, Children = { ok, cancel } },
                    },
                },
            };
            ok.Click += ( _, _ ) => window.DialogResult = true;
            window.Loaded += ( _, _ ) => {
                input.Focus();
                // Select the name but not the extension, like Explorer does.
                int dot = input.Text.LastIndexOf( '.' );
                input.Select( 0, dot > 0 ? dot : input.Text.Length );
            };
            return window.ShowDialog() == true ? input.Text : null;
        }
    }
}
