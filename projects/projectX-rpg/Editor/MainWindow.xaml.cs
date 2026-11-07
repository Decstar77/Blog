using System.ComponentModel;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using AvalonDock.Layout;
using AvalonDock.Layout.Serialization;
using AvalonDock.Themes;
using Editor.Core;
using Editor.Panels;
using Editor.Panels.AssetBrowser;
using Editor.Panels.Inspector;

namespace Editor {
    public partial class MainWindow : Window {
        private readonly AssetBrowserViewModel assetBrowser;
        private readonly InspectorPanel inspector = new();

        // ContentId -> panel content, so a deserialized layout can be re-attached to the live panels.
        private readonly Dictionary<string, object> panels = new();
        private string defaultLayout = "";

        public MainWindow() {
            InitializeComponent();
            DockManager.Theme = new Vs2013DarkTheme();

            ConsolePanel console = new ConsolePanel();   // first, so it captures log output from everything below
            Console.WriteLine( $"[Editor] Assets: {EditorPaths.AssetPath}" );
            Console.WriteLine( $"[Editor] Cooked: {EditorPaths.CookedPath}" );

            assetBrowser = new AssetBrowserViewModel( EditorPaths.AssetPath );
            assetBrowser.SelectionChanged += ( _, _ ) =>
                inspector.ShowAsset( assetBrowser.Selection.Count == 1 ? assetBrowser.Selection[ 0 ] : null, assetBrowser.Selection.Count );

            Attach( "assetBrowser", new AssetBrowserPanel( assetBrowser ) );
            Attach( "inspector",    inspector );
            Attach( "console",      console );
            Attach( "hierarchy",    Stub( "Scene hierarchy goes here." ) );
            Attach( "viewport",     Stub( "Viewport: host the OpenGL renderer here." ) );

            defaultLayout = SerializeLayout();
            Loaded += ( _, _ ) => LoadSavedLayout();
            Closing += OnClosing;
        }

        private void Attach( string contentId, object content ) {
            LayoutContent slot = DockManager.Layout.Descendents().OfType<LayoutContent>().Single( c => c.ContentId == contentId );
            slot.Content = content;
            panels[ contentId ] = content;
        }

        private static UIElement Stub( string text ) => new Border {
            Background = (Brush)Application.Current.FindResource( "PanelBrush" ),
            Child = new TextBlock {
                Text = text,
                Foreground = (Brush)Application.Current.FindResource( "TextDimBrush" ),
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
            },
        };

        // ---- Layout persistence ----

        private string SerializeLayout() {
            using StringWriter writer = new StringWriter();
            new XmlLayoutSerializer( DockManager ).Serialize( writer );
            return writer.ToString();
        }

        private bool DeserializeLayout( string xml ) {
            try {
                XmlLayoutSerializer serializer = new XmlLayoutSerializer( DockManager );
                serializer.LayoutSerializationCallback += ( _, e ) => {
                    if ( e.Model.ContentId != null && panels.TryGetValue( e.Model.ContentId, out object? content ) ) {
                        e.Content = content;
                    } else {
                        e.Cancel = true;   // a panel that no longer exists
                    }
                };
                using StringReader reader = new StringReader( xml );
                serializer.Deserialize( reader );

                // A saved layout from before a panel was added won't mention it; fall back rather than lose the panel.
                HashSet<string> present =AllContents().Select( c => c.ContentId ).ToHashSet();
                return panels.Keys.All( present.Contains );
            } catch ( Exception e ) {
                Console.Error.WriteLine( $"[Editor] Couldn't restore layout: {e.Message}" );
                return false;
            }
        }

        private void LoadSavedLayout() {
            if ( File.Exists( EditorPaths.LayoutFile ) && !DeserializeLayout( File.ReadAllText( EditorPaths.LayoutFile ) ) ) {
                ResetLayout();
            }
        }

        private void ResetLayout() => DeserializeLayout( defaultLayout );

        private void OnClosing( object? sender, CancelEventArgs e ) {
            try {
                Directory.CreateDirectory( Path.GetDirectoryName( EditorPaths.LayoutFile )! );
                File.WriteAllText( EditorPaths.LayoutFile, SerializeLayout() );
            } catch ( Exception ) {
                // Not worth blocking shutdown over.
            }
            assetBrowser.Dispose();
        }

        private IEnumerable<LayoutContent> AllContents() =>
            DockManager.Layout.Descendents().OfType<LayoutContent>().Concat( DockManager.Layout.Hidden ).Distinct();

        // ---- Menus ----

        private void WindowMenu_SubmenuOpened( object sender, RoutedEventArgs e ) {
            if ( e.OriginalSource != WindowMenu ) {
                return;
            }
            WindowMenu.Items.Clear();
            foreach ( LayoutAnchorable panel in AllContents().OfType<LayoutAnchorable>().OrderBy( a => a.Title ) ) {
                MenuItem item = new MenuItem { Header = panel.Title, IsCheckable = true, IsChecked = panel.IsVisible };
                item.Click += ( _, _ ) => {
                    if ( panel.IsVisible ) {
                        panel.Hide();
                    } else {
                        panel.Show();
                        panel.IsActive = true;
                    }
                };
                WindowMenu.Items.Add( item );
            }
            WindowMenu.Items.Add( new Separator() );
            MenuItem reset = new MenuItem { Header = "Reset layout" };
            reset.Click += ( _, _ ) => ResetLayout();
            WindowMenu.Items.Add( reset );
        }

        private void Exit_Click( object sender, RoutedEventArgs e ) => Close();
    }
}
