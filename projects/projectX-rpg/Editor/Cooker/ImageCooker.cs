namespace Editor {
    public static class ImageCooker {
        public static bool CookImage( string imageName ) {
            // Use STB image or whatever to load then save the raw data to disk in the engine format. 
            string osPath = Cooker.ToOsPath(imageName);

            return true;
        }
    }
}
