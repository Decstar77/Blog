using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Editor.Panels.AssetBrowser;

namespace Editor.Panels.Inspector {
    // Stub inspector: for now it only knows how to describe assets picked in the asset browser.
    public partial class InspectorPanel : UserControl {
        private int requestId;

        public InspectorPanel() {
            InitializeComponent();
        }

        public async void ShowAsset( AssetItem? item, int selectionCount = 1 ) {
            int id = ++requestId;
            Preview.Source = null;
            PreviewFrame.Visibility = Visibility.Collapsed;

            if ( item == null ) {
                Title.Text = selectionCount > 1 ? $"{selectionCount} assets selected" : "Nothing selected";
                Subtitle.Text = "";
                Rows.ItemsSource = null;
                return;
            }

            Title.Text = item.Name;
            Subtitle.Text = item.IsFolder ? "Folder" : $"{item.Kind} resource";

            List<KeyValuePair<string, string>> rows = new() {
                new( "Resource", item.ResourceName ),
                new( "Path", item.OsPath ),
                new( "Modified", item.Modified.ToString( "yyyy-MM-dd HH:mm" ) ),
            };
            if ( !item.IsFolder ) {
                rows.Insert( 1, new( "Size", AssetItem.FormatSize( item.SizeBytes ) ) );
            }
            Rows.ItemsSource = rows;

            if ( item.Kind == AssetKind.Image ) {
                var (header, image) = await Task.Run( () => (ThumbnailLoader.ReadHeader( item.OsPath ), ThumbnailLoader.Decode( item.OsPath, 512 )) );
                if ( id != requestId ) {
                    return;   // selection moved on while we were decoding
                }
                if ( header is { } h ) {
                    rows.Insert( 2, new( "Dimensions", $"{h.Width} × {h.Height}" ) );
                    rows.Insert( 3, new( "Format", h.Format.ToString() ) );
                    Rows.ItemsSource = null;
                    Rows.ItemsSource = rows;
                    // Pixel art stays crisp when shown small; photos look better smoothed.
                    RenderOptions.SetBitmapScalingMode( Preview, h.Width <= 256 ? BitmapScalingMode.NearestNeighbor : BitmapScalingMode.HighQuality );
                }
                if ( image != null ) {
                    Preview.Source = image;
                    PreviewFrame.Visibility = Visibility.Visible;
                }
            }
        }
    }
}
