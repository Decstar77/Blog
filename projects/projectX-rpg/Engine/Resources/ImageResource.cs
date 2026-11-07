using System.Runtime.InteropServices;

namespace Engine {
    public enum ImageFormat : int {
        Invalid = 0,
        RGB     = 1,
        RGBA    = 2,
        RGBA_F  = 3,
    }

    [StructLayout( LayoutKind.Sequential )]
    public struct ImageFileFormat {
        public const int    Magic = 0x474D4958; // "XIMG"

        public int          MagicNumber;
        public int          Width;
        public int          Height;
        public ImageFormat  Format;
        public int          Channels;

        public int ImageSizeBytes => Width * Height * FormatSizeBytes( Format );

        public static int FormatSizeBytes( ImageFormat format ) {
            switch ( format ) {
                case ImageFormat.Invalid:   return 0;
                case ImageFormat.RGB:       return 3;
                case ImageFormat.RGBA:      return 4;
                case ImageFormat.RGBA_F:    return 4 * 4;
                default:                    return 0;
            }
        }
    }

    public class ImageResource : Resource {
        public OpenGLTexture2D  GLTexture { get; private set; } = OpenGLTexture2D.InvalidTexture;
        public int              Width => GLTexture.Width;
        public int              Height => GLTexture.Height;

        public override bool Cook( string assetOsPath, string cookedOsPath ) => ImageCooker.Cook( assetOsPath, cookedOsPath );

        public override bool LoadFromSource( ResourceFile resourceFile ) {
            ImageFileFormat header = resourceFile.ReadStruct<ImageFileFormat>();
            if ( header.MagicNumber != ImageFileFormat.Magic ) {
                Console.Error.WriteLine( $"[ImageResource] {resourceFile.FileName}: bad magic number" );
                return false;
            }
            if ( header.Width <= 0 || header.Height <= 0 || header.ImageSizeBytes == 0 ) {
                Console.Error.WriteLine( $"[ImageResource] {resourceFile.FileName}: invalid header {header.Width}x{header.Height} {header.Format}" );
                return false;
            }

            ReadOnlySpan<byte> pixels = resourceFile.ReadBytes( header.ImageSizeBytes );
            GLTexture = OpenGLTexture2D.Create( OpenGLInstance.Get(), header.Width, header.Height, header.Format, pixels );
            return true;
        }

        public override void Dispose() {
            if ( GLTexture != OpenGLTexture2D.InvalidTexture ) {
                GLTexture.Dispose();
                GLTexture = OpenGLTexture2D.InvalidTexture;
            }
        }
    }
}
