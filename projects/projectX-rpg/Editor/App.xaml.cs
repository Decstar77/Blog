using System.Windows;
using Editor.Core;

namespace Editor {
    public partial class App : Application {
        protected override void OnStartup( StartupEventArgs e ) {
            base.OnStartup( e );
            EditorPaths.ParseArgs( e.Args );

            // Built-in Fluent dark styles for stock controls (menus, trees, scrollbars, text boxes).
            // Set here rather than in XAML so it merges alongside the palette in App.xaml instead of being replaced by it.
            ThemeMode = ThemeMode.Dark;

            MainWindow = new MainWindow();
            MainWindow.Show();
        }
    }
}
