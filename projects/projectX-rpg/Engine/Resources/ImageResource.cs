using Silk.NET.OpenGL;
using System.Runtime.InteropServices;

namespace Engine {
    enum ImageFormat : int { 
        Invalid = 0,
        RGB     = 1,
        RGBA    = 2,
        RGBA_F  = 3,
    }

    [StructLayout(LayoutKind.Sequential)]
    struct ImageFileFormat {
        public int          MagicNumber;
        public int          Width;
        public int          Height;
        public ImageFormat  Format;
        public int          Channels;

        public int ImageSizeBytes => Width * Height * FormatSizeBytes( Format );

        public int FormatSizeBytes( ImageFormat format ) {
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
        public OpenGLTexture2D GLTexture { get; private set; } = OpenGLTexture2D.InvalidTexture;

        public override bool LoadFromSource( ResourceFile resourceFile ) {
            //ImageFileFormat format =  resourceFile.ReadStruct<ImageFileFormat>();
            //byte[] data = resourceFile.ReadBytes( format.ImageSizeBytes );
            //GL gl =  OpenGLInstance.Get();
            //gl.CreateTexture()
            return true;
        }
    }
}
