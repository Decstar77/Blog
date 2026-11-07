using StbImageSharp;

namespace Engine {
    public static class ImageCooker {
        // Decodes a png/jpg/tga/bmp/... with stb_image and writes it as an ImageFileFormat header + raw
        // RGBA8 pixels (rows top to bottom), which is what the renderer expects.
        public static bool Cook( string assetOsPath, string cookedOsPath ) {
            try {
                ImageResult image;
                using ( FileStream stream = File.OpenRead( assetOsPath ) ) {
                    image = ImageResult.FromStream( stream, ColorComponents.RedGreenBlueAlpha );
                }

                ImageFileFormat header = new ImageFileFormat {
                    MagicNumber = ImageFileFormat.Magic,
                    Width       = image.Width,
                    Height      = image.Height,
                    Format      = ImageFormat.RGBA,
                    Channels    = 4,
                };
                ResourceFile.Write( cookedOsPath, header, image.Data );
                return true;
            } catch ( Exception e ) {
                Console.Error.WriteLine( $"[ImageCooker] Failed to cook '{assetOsPath}': {e.Message}" );
                return false;
            }
        }
    }
}
