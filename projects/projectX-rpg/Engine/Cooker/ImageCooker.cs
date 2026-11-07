using StbImageSharp;

namespace Engine {
    public static class ImageCooker {
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
