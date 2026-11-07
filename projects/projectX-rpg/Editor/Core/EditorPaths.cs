using Engine.Resources;

namespace Editor.Core {
    // Where the editor finds raw assets and writes cooked ones. Mirrors the paths Game.cs hands to ResourceManager.
    // Override on the command line: Editor.exe --project <dir> --assets <dir>
    public static class EditorPaths {
        public static string ProjectPath { get; private set; } = "C:/Projects/2025/Blog/projects/projectX-rpg";
        public static string AssetPath   { get; private set; } = "C:/Projects/Assets";
        public static string CookedPath => Path.GetFullPath( Path.Combine( ProjectPath, ResourceManager.CookedFolder ) );

        public static string LayoutFile => Path.Combine(
            Environment.GetFolderPath( Environment.SpecialFolder.LocalApplicationData ), "projectX-rpg", "editor-layout.xml" );

        public static void ParseArgs( string[] args ) {
            for ( int i = 0; i + 1 < args.Length; i++ ) {
                switch ( args[ i ] ) {
                    case "--project": ProjectPath = args[ ++i ]; break;
                    case "--assets":  AssetPath   = args[ ++i ]; break;
                }
            }
        }

        // The name ResourceManager.Load<T>() expects: relative to the asset root, forward slashes.
        public static string ToResourceName( string assetOsPath ) =>
            Path.GetRelativePath( AssetPath, assetOsPath ).Replace( '\\', '/' );

        // Same layout ResourceManager uses: Cooked/<lowercased resource name>.res
        public static string ToCookedOsPath( string assetOsPath ) =>
            Path.GetFullPath( Path.Combine( CookedPath, ToResourceName( assetOsPath ).ToLowerInvariant() + ResourceManager.CookedExtension ) );
    }
}
