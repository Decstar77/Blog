
namespace Editor {
    public static class Cooker {
        public static string RawAssetPath { get; set; } = "";

        public static string ToOsPath( string name ) => RawAssetPath + name;
    }
}
