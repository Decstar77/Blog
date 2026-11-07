using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Windows;
using System.Windows.Input;
using System.Windows.Threading;
using Editor.Core;
using Engine.Resources;
using Microsoft.Win32;

namespace Editor.Panels.AssetBrowser {
    // Browses <project>/Resources. Source files (png, ...) are never browsed in place: they're imported
    // (dropped or added via the dialog), cooked into the current folder as <name>.res, and only the .res is kept.
    public sealed class AssetBrowserViewModel : ObservableObject, IDisposable {
        private const int MaxSearchResults = 1000;

        private static readonly EnumerationOptions listOptions = new() { IgnoreInaccessible = true };
        private static readonly EnumerationOptions recurseOptions = new() { IgnoreInaccessible = true, RecurseSubdirectories = true };

        public string RootPath { get; }
        public ObservableCollection<FolderNode> Roots { get; } = new();
        public ObservableCollection<AssetItem> Items { get; } = new();

        private readonly Stack<string> history = new();
        private readonly Dispatcher dispatcher = Dispatcher.CurrentDispatcher;
        private readonly DispatcherTimer refreshDebounce;
        private readonly DispatcherTimer searchDebounce;
        private FileSystemWatcher? watcher;
        private CancellationTokenSource loadCancel = new();

        private string currentPath = "";
        public string CurrentPath { get => currentPath; private set { if ( Set( ref currentPath, value ) ) { OnPropertyChanged( nameof( Breadcrumb ) ); } } }
        public string Breadcrumb => DisplayPath( CurrentPath );

        private string searchText = "";
        public string SearchText {
            get => searchText;
            set { if ( Set( ref searchText, value ) ) { searchDebounce.Stop(); searchDebounce.Start(); } }
        }

        private double tileSize = 96;
        public double TileSize { get => tileSize; set => Set( ref tileSize, value ); }

        private string status = "";
        public string Status { get => status; private set => Set( ref status, value ); }

        private bool isBusy;
        public bool IsBusy { get => isBusy; private set => Set( ref isBusy, value ); }

        private IReadOnlyList<AssetItem> selection = Array.Empty<AssetItem>();
        public IReadOnlyList<AssetItem> Selection {
            get => selection;
            set { selection = value; SelectionChanged?.Invoke( this, EventArgs.Empty ); UpdateStatus(); }
        }
        public event EventHandler? SelectionChanged;

        // Raised when the browser moves to a new folder so the view can sync the tree.
        public event EventHandler? Navigated;

        public ICommand BackCommand { get; }
        public ICommand UpCommand { get; }
        public ICommand RefreshCommand { get; }
        public ICommand AddResourceCommand { get; }
        public ICommand NewFolderCommand { get; }
        public ICommand RenameCommand { get; }
        public ICommand DeleteCommand { get; }
        public ICommand ShowInExplorerCommand { get; }
        public ICommand CopyResourceNameCommand { get; }
        public ICommand ClearSearchCommand { get; }

        public AssetBrowserViewModel( string rootPath ) {
            RootPath = Path.GetFullPath( rootPath );
            Directory.CreateDirectory( RootPath );

            refreshDebounce = new DispatcherTimer( TimeSpan.FromMilliseconds( 300 ), DispatcherPriority.Background, ( _, _ ) => { refreshDebounce!.Stop(); Reload(); }, dispatcher );
            searchDebounce  = new DispatcherTimer( TimeSpan.FromMilliseconds( 250 ), DispatcherPriority.Background, ( _, _ ) => { searchDebounce!.Stop(); Reload(); }, dispatcher );
            refreshDebounce.Stop();
            searchDebounce.Stop();

            BackCommand             = new RelayCommand( GoBack, () => history.Count > 0 );
            UpCommand               = new RelayCommand( GoUp, () => !IsRoot( CurrentPath ) );
            RefreshCommand          = new RelayCommand( RefreshAll );
            AddResourceCommand      = new RelayCommand( p => AddResourceViaDialog( TargetFolderFrom( p ) ), _ => !IsBusy );
            NewFolderCommand        = new RelayCommand( p => NewFolder( TargetFolderFrom( p ) ) );
            RenameCommand           = new RelayCommand( p => Rename( ItemsFrom( p ).FirstOrDefault() ), p => ItemsFrom( p ).Count == 1 );
            DeleteCommand           = new RelayCommand( p => Delete( ItemsFrom( p ) ), p => ItemsFrom( p ).Any() );
            ShowInExplorerCommand   = new RelayCommand( p => ShowInExplorer( ItemsFrom( p ).FirstOrDefault()?.OsPath ?? CurrentPath ) );
            CopyResourceNameCommand = new RelayCommand( p => Clipboard.SetText( string.Join( Environment.NewLine, ItemsFrom( p ).Select( i => i.ResourceName ) ) ),
                                                        p => ItemsFrom( p ).Any() );
            ClearSearchCommand      = new RelayCommand( () => SearchText = "" );

            Roots.Add( new FolderNode( RootPath ) { IsExpanded = true } );
            Navigate( RootPath, addHistory: false );
        }

        // Commands take an explicit item, a list of items, or fall back to the current selection.
        private IReadOnlyList<AssetItem> ItemsFrom( object? parameter ) => parameter switch {
            AssetItem item                     => new[] { item },
            System.Collections.IList list      => list.OfType<AssetItem>().ToList(),
            _                                  => Selection,
        };

        // Folder-targeted commands (add, new folder) accept a folder item, a tree node, or default to the current folder.
        private string TargetFolderFrom( object? parameter ) => parameter switch {
            AssetItem { IsFolder: true } folder => folder.OsPath,
            FolderNode node                     => node.OsPath,
            string path when Directory.Exists( path ) => path,
            _                                   => CurrentPath,
        };

        public string DisplayPath( string osPath ) =>
            Path.GetRelativePath( Path.GetDirectoryName( RootPath ) ?? RootPath, osPath ).Replace( '\\', '/' );

        // ---- Navigation ----

        public void Navigate( string osPath, bool addHistory = true ) {
            osPath = Path.GetFullPath( osPath );
            if ( !Directory.Exists( osPath ) || !IsUnderRoot( osPath ) ) {
                return;
            }
            if ( string.Equals( osPath, CurrentPath, StringComparison.OrdinalIgnoreCase ) ) {
                return;
            }
            if ( addHistory && CurrentPath.Length > 0 ) {
                history.Push( CurrentPath );
            }
            CurrentPath = osPath;
            searchText = "";
            OnPropertyChanged( nameof( SearchText ) );
            Watch( osPath );
            Reload();
            Navigated?.Invoke( this, EventArgs.Empty );
        }

        public void Activate( AssetItem item ) {
            if ( item.IsFolder ) {
                Navigate( item.OsPath );
            }
        }

        private void GoBack() {
            while ( history.Count > 0 ) {
                string previous = history.Pop();
                if ( Directory.Exists( previous ) ) {   // skip folders deleted or renamed since
                    Navigate( previous, addHistory: false );
                    return;
                }
            }
        }

        private void GoUp() {
            string? parent = Path.GetDirectoryName( CurrentPath );
            if ( parent != null ) {
                Navigate( parent );
            }
        }

        private void RefreshAll() {
            foreach ( FolderNode root in Roots ) {
                root.InvalidateTree();
            }
            Reload();
            Navigated?.Invoke( this, EventArgs.Empty );
        }

        // ---- Listing ----

        // Re-lists the current folder (or runs the search) on a worker thread, then swaps the results in.
        private async void Reload() {
            loadCancel.Cancel();
            loadCancel = new CancellationTokenSource();
            CancellationToken cancel = loadCancel.Token;

            string folder = CurrentPath;
            string query = SearchText.Trim();

            List<AssetItem> results;
            try {
                results = await Task.Run( () => query.Length > 0 ? Search( folder, query, cancel ) : List( folder ), cancel );
            } catch ( OperationCanceledException ) {
                return;
            }
            if ( cancel.IsCancellationRequested ) {
                return;
            }

            Items.Clear();
            foreach ( AssetItem item in results ) {
                Items.Add( item );
                ThumbnailLoader.Request( item, cancel );
            }
            UpdateStatus( results.Count >= MaxSearchResults ? $" (first {MaxSearchResults})" : "" );
        }

        private static bool IsResourceFile( FileSystemInfo info ) =>
            info.Name.EndsWith( ResourceManager.Extension, StringComparison.OrdinalIgnoreCase );

        private static List<AssetItem> List( string folder ) {
            try {
                DirectoryInfo dir = new DirectoryInfo( folder );
                IEnumerable<AssetItem> folders = dir.EnumerateDirectories( "*", listOptions ).OrderBy( d => d.Name, StringComparer.OrdinalIgnoreCase ).Select( d => new AssetItem( d ) );
                IEnumerable<AssetItem> files   = dir.EnumerateFiles( "*", listOptions ).Where( IsResourceFile ).OrderBy( f => f.Name, StringComparer.OrdinalIgnoreCase ).Select( f => new AssetItem( f ) );
                return folders.Concat( files ).ToList();
            } catch ( IOException ) {
                return new List<AssetItem>();
            } catch ( UnauthorizedAccessException ) {
                return new List<AssetItem>();
            }
        }

        private static List<AssetItem> Search( string folder, string query, CancellationToken cancel ) {
            List<AssetItem> results = new();
            foreach ( FileSystemInfo info in new DirectoryInfo( folder ).EnumerateFileSystemInfos( "*", recurseOptions ) ) {
                cancel.ThrowIfCancellationRequested();
                if ( ( info is DirectoryInfo || IsResourceFile( info ) ) && info.Name.Contains( query, StringComparison.OrdinalIgnoreCase ) ) {
                    string location = Path.GetRelativePath( folder, Path.GetDirectoryName( info.FullName ) ?? folder ).Replace( '\\', '/' );
                    results.Add( new AssetItem( info ) { Location = location == "." ? null : location } );
                    if ( results.Count >= MaxSearchResults ) {
                        break;
                    }
                }
            }
            return results;
        }

        // ---- Import ----

        private void AddResourceViaDialog( string targetFolder ) {
            string supported = string.Join( ";", AssetKinds.SupportedExtensions.Select( e => "*" + e ) );
            OpenFileDialog dialog = new OpenFileDialog {
                Title = $"Add resources to {DisplayPath( targetFolder )}",
                Multiselect = true,
                Filter = $"Supported files ({supported})|{supported}|All files (*.*)|*.*",
            };
            if ( dialog.ShowDialog( Application.Current.MainWindow ) == true ) {
                _ = ImportAsync( dialog.FileNames, targetFolder );
            }
        }

        // Cooks each source file into targetFolder as "<file name>.res". Dropped folders keep their structure.
        // Files with no cooker are skipped and reported as not supported.
        public async Task ImportAsync( IReadOnlyList<string> sources, string? targetFolder = null ) {
            targetFolder ??= CurrentPath;
            if ( IsBusy ) {
                return;
            }

            List<(string Source, string Dest)> jobs = new();
            List<string> unsupported = new();
            foreach ( string source in sources ) {
                if ( Directory.Exists( source ) ) {
                    string parent = Path.GetDirectoryName( Path.GetFullPath( source ) ) ?? source;
                    foreach ( string file in Directory.EnumerateFiles( source, "*", recurseOptions ) ) {
                        string relDir = Path.GetRelativePath( parent, Path.GetDirectoryName( file )! );
                        Queue( file, Path.Combine( targetFolder, relDir ) );
                    }
                } else if ( File.Exists( source ) ) {
                    Queue( source, targetFolder );
                }
            }
            void Queue( string file, string folder ) {
                if ( AssetKinds.CreateCooker( file ) == null ) {
                    unsupported.Add( Path.GetFileName( file ) );
                } else {
                    jobs.Add( (file, EditorPaths.ToResourceOsPath( folder, Path.GetFileName( file ) )) );
                }
            }

            int failed = 0, replaced = 0;
            if ( jobs.Count > 0 ) {
                IsBusy = true;
                Stopwatch timer = Stopwatch.StartNew();
                await Task.Run( () => {
                    for ( int i = 0; i < jobs.Count; i++ ) {
                        (string source, string dest) = jobs[ i ];
                        string label = $"Cooking {i + 1}/{jobs.Count}: {Path.GetFileName( source )}";
                        dispatcher.BeginInvoke( () => Status = label );

                        if ( File.Exists( dest ) ) {
                            replaced++;
                        }
                        using Engine.Resource cooker = AssetKinds.CreateCooker( source )!;
                        if ( cooker.Cook( source, dest ) ) {
                            Console.WriteLine( $"[AssetBrowser] {source} -> {EditorPaths.ToResourceName( dest )}" );
                        } else {
                            failed++;
                        }
                    }
                } );
                IsBusy = false;

                string summary = $"Added {jobs.Count - failed} resource(s) to {DisplayPath( targetFolder )} in {timer.Elapsed.TotalSeconds:0.0}s"
                               + ( replaced > 0 ? $", {replaced} replaced" : "" )
                               + ( failed > 0 ? $", {failed} failed (see Console)" : "" );
                Console.WriteLine( $"[AssetBrowser] {summary}" );
                RefreshFolder( targetFolder );
                Status = summary;
            }

            if ( unsupported.Count > 0 ) {
                foreach ( string name in unsupported ) {
                    Console.Error.WriteLine( $"[AssetBrowser] Not supported (no cooker): {name}" );
                }
                const int shown = 12;
                string list = string.Join( Environment.NewLine, unsupported.Take( shown ).Select( n => "    " + n ) )
                            + ( unsupported.Count > shown ? $"{Environment.NewLine}    ...and {unsupported.Count - shown} more" : "" );
                MessageBox.Show( Application.Current.MainWindow,
                    $"These files aren't supported yet (no cooker for their type), so they were skipped:{Environment.NewLine}{Environment.NewLine}{list}"
                    + $"{Environment.NewLine}{Environment.NewLine}Supported: {string.Join( " ", AssetKinds.SupportedExtensions )}",
                    "Not supported", MessageBoxButton.OK, MessageBoxImage.Information );
            }
        }

        // ---- Organising ----

        private void NewFolder( string parent ) {
            string? name = PromptDialog.Ask( "New folder", $"Folder name in {DisplayPath( parent )}:", "New folder" );
            if ( string.IsNullOrWhiteSpace( name ) || !IsValidName( name ) ) {
                return;
            }
            string path = Path.Combine( parent, name.Trim() );
            if ( Directory.Exists( path ) ) {
                Status = $"'{name}' already exists";
                return;
            }
            Directory.CreateDirectory( path );
            RefreshFolder( parent );
        }

        private void Rename( AssetItem? item ) {
            if ( item == null ) {
                return;
            }
            string? name = PromptDialog.Ask( "Rename", item.IsFolder ? "Folder name:" : "Resource name (code that loads the old name will need updating):", item.Name );
            if ( string.IsNullOrWhiteSpace( name ) || name == item.Name || !IsValidName( name ) ) {
                return;
            }
            name = name.Trim();
            string parent = Path.GetDirectoryName( item.OsPath )!;
            try {
                if ( item.IsFolder ) {
                    Directory.Move( item.OsPath, Path.Combine( parent, name ) );
                } else {
                    File.Move( item.OsPath, Path.Combine( parent, name + ResourceManager.Extension ) );
                }
                Console.WriteLine( $"[AssetBrowser] Renamed {item.ResourceName} -> {name}" );
            } catch ( Exception e ) {
                MessageBox.Show( Application.Current.MainWindow, e.Message, "Rename failed", MessageBoxButton.OK, MessageBoxImage.Warning );
            }
            RefreshFolder( parent );
        }

        // Sends to the Recycle Bin rather than deleting outright, since Resources holds the only cooked copy.
        private void Delete( IReadOnlyList<AssetItem> items ) {
            if ( items.Count == 0 ) {
                return;
            }
            string what = items.Count == 1 ? $"'{items[ 0 ].Name}'" : $"{items.Count} items";
            if ( MessageBox.Show( Application.Current.MainWindow, $"Move {what} to the Recycle Bin?", "Delete",
                                  MessageBoxButton.OKCancel, MessageBoxImage.Question ) != MessageBoxResult.OK ) {
                return;
            }
            foreach ( AssetItem item in items ) {
                try {
                    if ( item.IsFolder ) {
                        Microsoft.VisualBasic.FileIO.FileSystem.DeleteDirectory( item.OsPath,
                            Microsoft.VisualBasic.FileIO.UIOption.OnlyErrorDialogs, Microsoft.VisualBasic.FileIO.RecycleOption.SendToRecycleBin );
                    } else {
                        Microsoft.VisualBasic.FileIO.FileSystem.DeleteFile( item.OsPath,
                            Microsoft.VisualBasic.FileIO.UIOption.OnlyErrorDialogs, Microsoft.VisualBasic.FileIO.RecycleOption.SendToRecycleBin );
                    }
                    Console.WriteLine( $"[AssetBrowser] Deleted {item.ResourceName}" );
                } catch ( Exception e ) {
                    Console.Error.WriteLine( $"[AssetBrowser] Couldn't delete {item.ResourceName}: {e.Message}" );
                }
            }
            foreach ( string parent in items.Select( i => Path.GetDirectoryName( i.OsPath )! ).Distinct() ) {
                RefreshFolder( parent );
            }
        }

        // True if paths can be moved into targetFolder: not into itself or its own subfolder, and not where they already are.
        public bool CanMove( IReadOnlyList<string> paths, string targetFolder ) {
            if ( IsBusy || paths.Count == 0 || !Directory.Exists( targetFolder ) || !IsUnderRoot( targetFolder ) ) {
                return false;
            }
            string target = Path.GetFullPath( targetFolder );
            bool anyElsewhere = false;
            foreach ( string path in paths ) {
                string full = Path.GetFullPath( path );
                if ( Directory.Exists( full ) && IsSameOrInside( target, full ) ) {
                    return false;
                }
                if ( !string.Equals( Path.GetDirectoryName( full ), target, StringComparison.OrdinalIgnoreCase ) ) {
                    anyElsewhere = true;
                }
            }
            return anyElsewhere;
        }

        // Moves resources/folders into targetFolder. Name clashes are skipped and reported rather than overwritten.
        public void Move( IReadOnlyList<string> paths, string targetFolder ) {
            if ( !CanMove( paths, targetFolder ) ) {
                return;
            }
            int moved = 0;
            List<string> clashes = new();
            HashSet<string> touched = new( StringComparer.OrdinalIgnoreCase ) { targetFolder };
            foreach ( string path in paths ) {
                string parent = Path.GetDirectoryName( path )!;
                if ( string.Equals( parent, targetFolder, StringComparison.OrdinalIgnoreCase ) ) {
                    continue;
                }
                string dest = Path.Combine( targetFolder, Path.GetFileName( path ) );
                if ( File.Exists( dest ) || Directory.Exists( dest ) ) {
                    clashes.Add( Path.GetFileName( path ) );
                    continue;
                }
                try {
                    string oldName = EditorPaths.ToResourceName( path );
                    if ( Directory.Exists( path ) ) {
                        Directory.Move( path, dest );
                    } else {
                        File.Move( path, dest );
                    }
                    Console.WriteLine( $"[AssetBrowser] Moved {oldName} -> {EditorPaths.ToResourceName( dest )}" );
                    touched.Add( parent );
                    moved++;
                } catch ( Exception e ) {
                    Console.Error.WriteLine( $"[AssetBrowser] Couldn't move {path}: {e.Message}" );
                }
            }
            foreach ( string folder in touched ) {
                RefreshFolder( folder );
            }
            Status = $"Moved {moved} item(s) to {DisplayPath( targetFolder )}"
                   + ( clashes.Count > 0 ? $", {clashes.Count} skipped (name already exists there)" : "" );
            if ( clashes.Count > 0 ) {
                MessageBox.Show( Application.Current.MainWindow,
                    $"{DisplayPath( targetFolder )} already has items with these names, so they weren't moved:{Environment.NewLine}{Environment.NewLine}"
                    + string.Join( Environment.NewLine, clashes.Take( 12 ).Select( n => "    " + n ) ),
                    "Move", MessageBoxButton.OK, MessageBoxImage.Information );
            }
        }

        private static bool IsSameOrInside( string path, string folder ) {
            string rel = Path.GetRelativePath( folder, path );
            return rel == "." || ( !rel.StartsWith( ".." ) && !Path.IsPathRooted( rel ) );
        }

        private static bool IsValidName( string name ) {
            if ( name.IndexOfAny( Path.GetInvalidFileNameChars() ) >= 0 ) {
                MessageBox.Show( Application.Current.MainWindow, $"'{name}' contains characters that can't be used in a file name.", "Invalid name",
                                 MessageBoxButton.OK, MessageBoxImage.Warning );
                return false;
            }
            return true;
        }

        // Picks up changes under a folder: re-reads its tree node (and ancestors, in case new subfolders appeared) and the listing.
        private void RefreshFolder( string folder ) {
            for ( string? path = folder; path != null && IsUnderRoot( path ); path = Path.GetDirectoryName( path ) ) {
                Roots.FirstOrDefault()?.Find( path )?.Invalidate();
            }
            if ( !Directory.Exists( CurrentPath ) ) {
                GoUp();
            }
            Reload();
            Navigated?.Invoke( this, EventArgs.Empty );
        }

        private static void ShowInExplorer( string osPath ) {
            string args = File.Exists( osPath ) ? $"/select,\"{osPath}\"" : $"\"{osPath}\"";
            Process.Start( new ProcessStartInfo( "explorer.exe", args ) { UseShellExecute = true } );
        }

        private void Watch( string folder ) {
            watcher?.Dispose();
            try {
                watcher = new FileSystemWatcher( folder ) { IncludeSubdirectories = false, EnableRaisingEvents = true };
                FileSystemEventHandler changed = ( _, _ ) => dispatcher.BeginInvoke( () => { refreshDebounce.Stop(); refreshDebounce.Start(); } );
                watcher.Created += changed;
                watcher.Deleted += changed;
                watcher.Changed += changed;
                watcher.Renamed += ( s, e ) => changed( s, e );
            } catch ( Exception ) {
                watcher = null;
            }
        }

        private void UpdateStatus( string suffix = "" ) {
            int folders = Items.Count( i => i.IsFolder );
            int files = Items.Count - folders;
            string text = SearchText.Length > 0
                ? $"{Items.Count} match(es) for \"{SearchText}\"{suffix}"
                : $"{folders} folder(s), {files} resource(s)";
            if ( Selection.Count == 1 ) {
                text += $"  ·  {Selection[ 0 ].ResourceName}";
            } else if ( Selection.Count > 1 ) {
                text += $"  ·  {Selection.Count} selected";
            }
            Status = text;
        }

        private bool IsRoot( string path ) => string.Equals( Path.GetFullPath( path ), RootPath, StringComparison.OrdinalIgnoreCase );
        public bool IsUnderRoot( string path ) => !Path.GetRelativePath( RootPath, Path.GetFullPath( path ) ).StartsWith( ".." );

        public void Dispose() {
            loadCancel.Cancel();
            watcher?.Dispose();
        }
    }
}
