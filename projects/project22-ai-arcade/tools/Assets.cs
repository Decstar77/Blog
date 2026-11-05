namespace arcade {
    // Assets are referenced in place from the shared asset library, not copied into the project.
    // An asset name is its path relative to the base folder, e.g. "Other/gunsoundpack/.../gun_pistol_shot_01.wav".
    public static class Assets {
        public static string AudioBasePath = @"C:\Projects\Assets\Audio";

        public static string AudioPath( string assetName ) => Path.Combine( AudioBasePath, assetName );
    }
}
