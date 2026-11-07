using Engine.Resources;

namespace Editor.Core {
    // Where the editor reads and writes cooked resources. Mirrors the base path Game.cs hands to ResourceManager.
    // Override on the command line: Editor.exe --project <dir>
    public static class EditorPaths {
        public static string ProjectPath { get; private set; } = "C:/Projects/2025/Blog/projects/projectX-rpg";
        public static string ResourcesPath => Path.GetFullPath( Path.Combine( ProjectPath, ResourceManager.ResourcesFolder ) );

        public static string LayoutFile => Path.Combine(
            Environment.GetFolderPath( Environment.SpecialFolder.LocalApplicationData ), "projectX-rpg", "editor-layout.xml" );

        public static void ParseArgs( string[] args ) {
            for ( int i = 0; i + 1 < args.Length; i++ ) {
                if ( args[ i ] == "--project" ) {
                    ProjectPath = args[ ++i ];
                }
            }
        }

        // "<Resources>/Textures/Sky.png.res" -> "Textures/Sky.png", the name ResourceManager.Load<T>() expects.
        public static string ToResourceName( string osPath ) {
            string rel = Path.GetRelativePath( ResourcesPath, osPath ).Replace( '\\', '/' );
            return rel.EndsWith( ResourceManager.Extension, StringComparison.OrdinalIgnoreCase ) ? rel[ ..^ResourceManager.Extension.Length ] : rel;
        }

        // Where a source file dropped into resourceFolder gets cooked to: <folder>/<file name with extension>.res
        public static string ToResourceOsPath( string resourceFolder, string sourceFileName ) =>
            Path.Combine( resourceFolder, sourceFileName + ResourceManager.Extension );
    }
}
