using System.Windows.Media;
using Editor.Core;
using Engine;
using Engine.Resources;

namespace Editor.Panels.AssetBrowser {
    public enum AssetKind { Folder, Image, Audio, Model, Archive, Other }

    public static class AssetKinds {
        private static readonly Dictionary<string, AssetKind> byExtension = new( StringComparer.OrdinalIgnoreCase ) {
            [ ".png" ] = AssetKind.Image, [ ".jpg" ] = AssetKind.Image, [ ".jpeg" ] = AssetKind.Image,
            [ ".bmp" ] = AssetKind.Image, [ ".tga" ] = AssetKind.Image, [ ".gif" ] = AssetKind.Image,
            [ ".psd" ] = AssetKind.Image, [ ".hdr" ] = AssetKind.Image,
            [ ".wav" ] = AssetKind.Audio, [ ".ogg" ] = AssetKind.Audio, [ ".mp3" ] = AssetKind.Audio, [ ".flac" ] = AssetKind.Audio,
            [ ".fbx" ] = AssetKind.Model, [ ".obj" ] = AssetKind.Model, [ ".gltf" ] = AssetKind.Model,
            [ ".glb" ] = AssetKind.Model, [ ".blend" ] = AssetKind.Model,
            [ ".zip" ] = AssetKind.Archive, [ ".rar" ] = AssetKind.Archive, [ ".7z" ] = AssetKind.Archive,
            [ ".unitypackage" ] = AssetKind.Archive,
        };

        // Which engine Resource cooks each source extension. Add entries here as new cookers land;
        // anything not listed is rejected as "not supported" on import.
        private static readonly Dictionary<string, Func<Resource>> cookers = new( StringComparer.OrdinalIgnoreCase ) {
            [ ".png" ]  = () => new ImageResource(),
            [ ".jpg" ]  = () => new ImageResource(),
            [ ".jpeg" ] = () => new ImageResource(),
            [ ".bmp" ]  = () => new ImageResource(),
            [ ".tga" ]  = () => new ImageResource(),
            [ ".gif" ]  = () => new ImageResource(),
            [ ".psd" ]  = () => new ImageResource(),
        };

        public static IEnumerable<string> SupportedExtensions => cookers.Keys;

        // Works on both source files ("a.png") and resource names ("a.png" from "a.png.res").
        public static AssetKind FromName( string name ) =>
            byExtension.TryGetValue( Path.GetExtension( name ), out AssetKind kind ) ? kind : AssetKind.Other;

        public static Resource? CreateCooker( string sourcePath ) =>
            cookers.TryGetValue( Path.GetExtension( sourcePath ), out var make ) ? make() : null;
    }

    // A folder or a cooked .res file under the Resources folder.
    public sealed class AssetItem : ObservableObject {
        public string       Name { get; }       // display name: file name minus ".res"
        public string       OsPath { get; }     // the .res file (or folder) on disk
        public AssetKind    Kind { get; }
        public bool         IsFolder => Kind == AssetKind.Folder;
        public string       Extension => IsFolder ? "" : Path.GetExtension( Name ).TrimStart( '.' ).ToUpperInvariant();
        public long         SizeBytes { get; }
        public DateTime     Modified { get; }
        public string       ResourceName => EditorPaths.ToResourceName( OsPath );

        // Set when the item came from a recursive search, so the tile can show where it lives.
        public string?      Location { get; init; }

        public string Glyph => Kind switch {
            AssetKind.Folder  => "",
            AssetKind.Image   => "",
            AssetKind.Audio   => "",
            AssetKind.Model   => "",
            AssetKind.Archive => "",
            _                 => "",
        };

        private ImageSource? thumbnail;
        public ImageSource? Thumbnail { get => thumbnail; set => Set( ref thumbnail, value ); }

        public AssetItem( FileSystemInfo info ) {
            OsPath   = info.FullName;
            Modified = info.LastWriteTime;
            if ( info is FileInfo file ) {
                Name      = info.Name.EndsWith( ResourceManager.Extension, StringComparison.OrdinalIgnoreCase )
                            ? info.Name[ ..^ResourceManager.Extension.Length ] : info.Name;
                Kind      = AssetKinds.FromName( Name );
                SizeBytes = file.Length;
            } else {
                Name = info.Name;
                Kind = AssetKind.Folder;
            }
        }

        public override string ToString() => Name;

        public static string FormatSize( long bytes ) => bytes switch {
            < 1024          => $"{bytes} B",
            < 1024 * 1024   => $"{bytes / 1024.0:0.#} KB",
            _               => $"{bytes / ( 1024.0 * 1024.0 ):0.#} MB",
        };
    }
}
