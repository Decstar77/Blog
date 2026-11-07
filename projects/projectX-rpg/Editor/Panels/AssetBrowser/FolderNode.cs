using System.Collections.ObjectModel;
using Editor.Core;

namespace Editor.Panels.AssetBrowser {
    // One folder in the asset browser's tree. Children are enumerated the first time the node is expanded.
    public sealed class FolderNode : ObservableObject {
        private static readonly FolderNode placeholder = new FolderNode();
        private static readonly EnumerationOptions dirOptions = new() { IgnoreInaccessible = true };

        public string Name { get; } = "";
        public string OsPath { get; } = "";
        public ObservableCollection<FolderNode> Children { get; } = new();

        private bool childrenLoaded;

        private bool isExpanded;
        public bool IsExpanded {
            get => isExpanded;
            set {
                if ( Set( ref isExpanded, value ) && value ) {
                    EnsureChildren();
                }
            }
        }

        private bool isSelected;
        public bool IsSelected { get => isSelected; set => Set( ref isSelected, value ); }

        private FolderNode() { }

        public FolderNode( string osPath, string? displayName = null ) {
            OsPath = Path.GetFullPath( osPath );
            Name = displayName ?? Path.GetFileName( OsPath );
            if ( HasSubfolders( OsPath ) ) {
                Children.Add( placeholder );   // gives the TreeView an expander arrow before we've looked inside
            }
        }

        public void EnsureChildren() {
            if ( childrenLoaded ) {
                return;
            }
            childrenLoaded = true;
            Children.Clear();
            try {
                foreach ( string dir in Directory.EnumerateDirectories( OsPath, "*", dirOptions ).OrderBy( d => d, StringComparer.OrdinalIgnoreCase ) ) {
                    Children.Add( new FolderNode( dir ) );
                }
            } catch ( IOException ) { } catch ( UnauthorizedAccessException ) { }
        }

        // Forget cached children so the next expand re-reads the disk.
        public void Invalidate() {
            if ( !childrenLoaded ) {
                return;
            }
            childrenLoaded = false;
            Children.Clear();
            if ( HasSubfolders( OsPath ) ) {
                Children.Add( placeholder );
            }
            if ( IsExpanded ) {
                EnsureChildren();
            }
        }

        // Expands down from this node to osPath and returns the node for it, or null if it's not underneath.
        public FolderNode? Reveal( string osPath ) {
            string rel = Path.GetRelativePath( OsPath, Path.GetFullPath( osPath ) );
            if ( rel == "." ) {
                return this;
            }
            if ( rel.StartsWith( ".." ) || Path.IsPathRooted( rel ) ) {
                return null;
            }
            FolderNode node = this;
            foreach ( string part in rel.Split( Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries ) ) {
                node.IsExpanded = true;
                FolderNode? next = node.Children.FirstOrDefault( c => string.Equals( c.Name, part, StringComparison.OrdinalIgnoreCase ) );
                if ( next == null ) {
                    return node;
                }
                node = next;
            }
            return node;
        }

        public override string ToString() => Name;   // also what screen readers / UI automation see

        private static bool HasSubfolders( string path ) {
            try {
                return Directory.EnumerateDirectories( path, "*", dirOptions ).Any();
            } catch ( IOException ) { return false; } catch ( UnauthorizedAccessException ) { return false; }
        }
    }
}
