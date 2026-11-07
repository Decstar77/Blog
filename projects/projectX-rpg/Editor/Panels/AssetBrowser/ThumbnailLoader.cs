using System.Collections.Concurrent;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Editor.Panels.AssetBrowser {
    // Decodes small thumbnails off the UI thread, a few at a time, and caches them by path + timestamp.
    public static class ThumbnailLoader {
        public const int DecodeWidth = 200;   // matches the tile size slider's max

        private static readonly SemaphoreSlim gate = new( 4 );
        private static readonly ConcurrentDictionary<(string, DateTime), ImageSource> cache = new();

        public static void Request( AssetItem item, CancellationToken cancel ) {
            if ( item.Kind != AssetKind.Image ) {
                return;
            }
            if ( cache.TryGetValue( (item.OsPath, item.Modified), out ImageSource? hit ) ) {
                item.Thumbnail = hit;
                return;
            }
            _ = Task.Run( async () => {
                await gate.WaitAsync( cancel ).ConfigureAwait( false );
                try {
                    cancel.ThrowIfCancellationRequested();
                    ImageSource? image = Decode( item.OsPath, DecodeWidth );
                    if ( image != null ) {
                        cache[ (item.OsPath, item.Modified) ] = image;
                        item.Thumbnail = image;   // frozen, so safe to hand across threads; WPF marshals the PropertyChanged
                    }
                } finally {
                    gate.Release();
                }
            }, cancel );
        }

        // Decodes at most maxWidth pixels wide (never upscales, so small sprites stay crisp).
        // Returns null for formats WIC can't read (tga, psd, hdr), which then just show their type glyph.
        public static ImageSource? Decode( string osPath, int maxWidth ) {
            try {
                using FileStream stream = File.OpenRead( osPath );
                int width = BitmapDecoder.Create( stream, BitmapCreateOptions.DelayCreation, BitmapCacheOption.None ).Frames[ 0 ].PixelWidth;
                stream.Position = 0;

                BitmapImage image = new BitmapImage();
                image.BeginInit();
                image.CacheOption = BitmapCacheOption.OnLoad;
                if ( width > maxWidth ) {
                    image.DecodePixelWidth = maxWidth;
                }
                image.StreamSource = stream;
                image.EndInit();
                image.Freeze();
                return image;
            } catch ( Exception ) {
                return null;
            }
        }

        // Reads only the header, for showing dimensions without decoding pixels.
        public static (int Width, int Height)? ReadSize( string osPath ) {
            try {
                using FileStream stream = File.OpenRead( osPath );
                BitmapFrame frame = BitmapDecoder.Create( stream, BitmapCreateOptions.DelayCreation, BitmapCacheOption.None ).Frames[ 0 ];
                return (frame.PixelWidth, frame.PixelHeight);
            } catch ( Exception ) {
                return null;
            }
        }
    }
}
