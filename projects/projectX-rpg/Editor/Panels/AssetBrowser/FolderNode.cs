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

        // Re-reads this folder's subfolders from disk. Existing child nodes are kept (only added/removed folders change),
        // so expansion and the tree's selection survive; clearing them would bounce the TreeView's selection to the root.
        public void Invalidate() {
            if ( !childrenLoaded ) {
                bool has = HasSubfolders( OsPath );
                if ( has && Children.Count == 0 ) {
                    Children.Add( placeholder );
                } else if ( !has ) {
                    Children.Clear();
                }
                return;
            }

            List<string> dirs;
            try {
                dirs = Directory.EnumerateDirectories( OsPath, "*", dirOptions ).OrderBy( d => d, StringComparer.OrdinalIgnoreCase ).ToList();
            } catch ( IOException ) {
                dirs = new();
            } catch ( UnauthorizedAccessException ) {
                dirs = new();
            }

            for ( int i = Children.Count - 1; i >= 0; i-- ) {
                if ( !dirs.Contains( Children[ i ].OsPath, StringComparer.OrdinalIgnoreCase ) ) {
                    Children.RemoveAt( i );
                }
            }
            for ( int i = 0; i < dirs.Count; i++ ) {
                int existing = IndexOf( dirs[ i ] );
                if ( existing == i ) {
                    continue;
                }
                if ( existing >= 0 ) {
                    Children.Move( existing, i );
                } else {
                    Children.Insert( i, new FolderNode( dirs[ i ] ) );
                }
            }
        }

        // Invalidate this node and every loaded descendant (full refresh).
        public void InvalidateTree() {
            Invalidate();
            if ( childrenLoaded ) {
                foreach ( FolderNode child in Children ) {
                    child.InvalidateTree();
                }
            }
        }

        private int IndexOf( string osPath ) {
            for ( int i = 0; i < Children.Count; i++ ) {
                if ( string.Equals( Children[ i ].OsPath, osPath, StringComparison.OrdinalIgnoreCase ) ) {
                    return i;
                }
            }
            return -1;
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

        // Finds an already-loaded node for osPath without expanding anything. Null if it isn't loaded.
        public FolderNode? Find( string osPath ) {
            if ( string.Equals( OsPath, Path.GetFullPath( osPath ), StringComparison.OrdinalIgnoreCase ) ) {
                return this;
            }
            if ( !childrenLoaded ) {
                return null;
            }
            foreach ( FolderNode child in Children ) {
                if ( child.Find( osPath ) is { } found ) {
                    return found;
                }
            }
            return null;
        }

        public override string ToString() => Name;   // also what screen readers / UI automation see

        private static bool HasSubfolders( string path ) {
            try {
                return Directory.EnumerateDirectories( path, "*", dirOptions ).Any();
            } catch ( IOException ) { return false; } catch ( UnauthorizedAccessException ) { return false; }
        }
    }
}
