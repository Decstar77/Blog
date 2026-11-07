using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Engine;

namespace Editor.Panels.AssetBrowser {
    // Builds thumbnails from cooked image resources off the UI thread, a few at a time, cached by path + timestamp.
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

        // Reads only the .res header.
        public static ImageFileFormat? ReadHeader( string resOsPath ) {
            try {
                using FileStream stream = File.OpenRead( resOsPath );
                Span<byte> bytes = stackalloc byte[ Marshal.SizeOf<ImageFileFormat>() ];
                stream.ReadExactly( bytes );
                ImageFileFormat header = MemoryMarshal.Read<ImageFileFormat>( bytes );
                return header.MagicNumber == ImageFileFormat.Magic ? header : null;
            } catch ( Exception ) {
                return null;
            }
        }

        // Decodes a cooked image resource, scaled down to at most maxWidth (never up, so small sprites stay crisp).
        public static ImageSource? Decode( string resOsPath, int maxWidth ) {
            try {
                byte[] data = File.ReadAllBytes( resOsPath );
                int headerSize = Marshal.SizeOf<ImageFileFormat>();
                ImageFileFormat header = MemoryMarshal.Read<ImageFileFormat>( data );
                if ( header.MagicNumber != ImageFileFormat.Magic || data.Length < headerSize + header.ImageSizeBytes ) {
                    return null;
                }

                int w = header.Width, h = header.Height;
                byte[] bgra = new byte[ w * h * 4 ];
                ReadOnlySpan<byte> src = data.AsSpan( headerSize, header.ImageSizeBytes );
                switch ( header.Format ) {
                    case ImageFormat.RGBA:
                        for ( int i = 0; i < bgra.Length; i += 4 ) {
                            bgra[ i ] = src[ i + 2 ]; bgra[ i + 1 ] = src[ i + 1 ]; bgra[ i + 2 ] = src[ i ]; bgra[ i + 3 ] = src[ i + 3 ];
                        }
                        break;
                    case ImageFormat.RGB:
                        for ( int i = 0, j = 0; i < bgra.Length; i += 4, j += 3 ) {
                            bgra[ i ] = src[ j + 2 ]; bgra[ i + 1 ] = src[ j + 1 ]; bgra[ i + 2 ] = src[ j ]; bgra[ i + 3 ] = 255;
                        }
                        break;
                    default:
                        return null;   // float formats: no preview yet
                }

                BitmapSource image = BitmapSource.Create( w, h, 96, 96, PixelFormats.Bgra32, null, bgra, w * 4 );
                if ( w > maxWidth ) {
                    double scale = (double)maxWidth / w;
                    // Copy into a WriteableBitmap so the full-size source isn't kept alive by the transform.
                    image = new WriteableBitmap( new TransformedBitmap( image, new ScaleTransform( scale, scale ) ) );
                }
                image.Freeze();
                return image;
            } catch ( Exception ) {
                return null;
            }
        }
    }
}
