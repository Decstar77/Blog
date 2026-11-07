using System.Windows.Media;
using Editor.Core;
using Engine;

namespace Editor.Panels.AssetBrowser {
    public enum AssetKind { Folder, Image, Audio, Model, Archive, Other }

    public enum CookState {
        NotCookable,    // no cooker for this kind yet
        Uncooked,       // cookable, no .res on disk
        Stale,          // .res is older than the source asset
        Cooked,         // .res is up to date
    }

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

        // Which engine Resource knows how to cook each kind. Add an entry here when a new cooker lands.
        private static readonly Dictionary<AssetKind, Func<Resource>> cookers = new() {
            [ AssetKind.Image ] = () => new ImageResource(),
        };

        public static AssetKind FromPath( string osPath ) =>
            byExtension.TryGetValue( Path.GetExtension( osPath ), out AssetKind kind ) ? kind : AssetKind.Other;

        public static Resource? CreateCooker( AssetKind kind ) => cookers.TryGetValue( kind, out var make ) ? make() : null;
        public static bool IsCookable( AssetKind kind ) => cookers.ContainsKey( kind );
    }

    public sealed class AssetItem : ObservableObject {
        public string       Name { get; }
        public string       OsPath { get; }
        public AssetKind    Kind { get; }
        public bool         IsFolder => Kind == AssetKind.Folder;
        public string       Extension => IsFolder ? "" : Path.GetExtension( OsPath ).TrimStart( '.' ).ToUpperInvariant();
        public long         SizeBytes { get; }
        public DateTime     Modified { get; }
        public string       ResourceName => EditorPaths.ToResourceName( OsPath );
        public string       CookedOsPath => EditorPaths.ToCookedOsPath( OsPath );

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

        private CookState cookState;
        public CookState CookState { get => cookState; private set => Set( ref cookState, value ); }

        private ImageSource? thumbnail;
        public ImageSource? Thumbnail { get => thumbnail; set => Set( ref thumbnail, value ); }

        public AssetItem( FileSystemInfo info ) {
            Name     = info.Name;
            OsPath   = info.FullName;
            Kind     = info is DirectoryInfo ? AssetKind.Folder : AssetKinds.FromPath( info.FullName );
            SizeBytes = info is FileInfo file ? file.Length : 0;
            Modified = info.LastWriteTime;
            RefreshCookState();
        }

        public void RefreshCookState() {
            if ( !AssetKinds.IsCookable( Kind ) ) {
                CookState = CookState.NotCookable;
                return;
            }
            FileInfo cooked = new FileInfo( CookedOsPath );
            if ( !cooked.Exists ) {
                CookState = CookState.Uncooked;
            } else {
                CookState = cooked.LastWriteTime >= Modified ? CookState.Cooked : CookState.Stale;
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
