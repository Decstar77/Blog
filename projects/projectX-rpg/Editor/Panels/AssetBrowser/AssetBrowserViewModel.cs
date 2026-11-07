using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Windows;
using System.Windows.Input;
using System.Windows.Threading;
using Editor.Core;

namespace Editor.Panels.AssetBrowser {
    public sealed class AssetBrowserViewModel : ObservableObject, IDisposable {
        private const int MaxSearchResults = 1000;

        private static readonly EnumerationOptions listOptions = new() { IgnoreInaccessible = true };
        private static readonly EnumerationOptions searchOptions = new() { IgnoreInaccessible = true, RecurseSubdirectories = true };

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
        public string Breadcrumb => Path.GetRelativePath( Path.GetDirectoryName( RootPath ) ?? RootPath, CurrentPath ).Replace( '\\', '/' );

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
        public ICommand CookCommand { get; }
        public ICommand OpenCommand { get; }
        public ICommand ShowInExplorerCommand { get; }
        public ICommand CopyResourceNameCommand { get; }
        public ICommand ClearSearchCommand { get; }

        public AssetBrowserViewModel( string rootPath ) {
            RootPath = Path.GetFullPath( rootPath );

            refreshDebounce = new DispatcherTimer( TimeSpan.FromMilliseconds( 300 ), DispatcherPriority.Background, ( _, _ ) => { refreshDebounce!.Stop(); Reload(); }, dispatcher );
            searchDebounce  = new DispatcherTimer( TimeSpan.FromMilliseconds( 250 ), DispatcherPriority.Background, ( _, _ ) => { searchDebounce!.Stop(); Reload(); }, dispatcher );
            refreshDebounce.Stop();
            searchDebounce.Stop();

            BackCommand             = new RelayCommand( GoBack, () => history.Count > 0 );
            UpCommand               = new RelayCommand( GoUp, () => !IsRoot( CurrentPath ) );
            RefreshCommand          = new RelayCommand( RefreshAll );
            CookCommand             = new RelayCommand( p => _ = CookAsync( ItemsFrom( p ) ), p => !IsBusy && ItemsFrom( p ).Any( i => i.IsFolder || AssetKinds.IsCookable( i.Kind ) ) );
            OpenCommand             = new RelayCommand( p => Open( ItemsFrom( p ).FirstOrDefault() ) );
            ShowInExplorerCommand   = new RelayCommand( p => ShowInExplorer( ItemsFrom( p ).FirstOrDefault()?.OsPath ?? CurrentPath ) );
            CopyResourceNameCommand = new RelayCommand( p => Clipboard.SetText( string.Join( Environment.NewLine, ItemsFrom( p ).Select( i => i.ResourceName ) ) ),
                                                        p => ItemsFrom( p ).Any() );
            ClearSearchCommand      = new RelayCommand( () => SearchText = "" );

            if ( Directory.Exists( RootPath ) ) {
                Roots.Add( new FolderNode( RootPath ) { IsExpanded = true } );
                Navigate( RootPath, addHistory: false );
            } else {
                CurrentPath = RootPath;
                Status = $"Asset folder not found: {RootPath}";
            }
        }

        // Commands take an explicit item, a list of items, or fall back to the current selection.
        private IReadOnlyList<AssetItem> ItemsFrom( object? parameter ) => parameter switch {
            AssetItem item                     => new[] { item },
            System.Collections.IList list      => list.OfType<AssetItem>().ToList(),
            _                                  => Selection,
        };

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
            } else if ( AssetKinds.IsCookable( item.Kind ) ) {
                _ = CookAsync( new[] { item } );
            } else {
                Open( item );
            }
        }

        private void GoBack() {
            if ( history.Count > 0 ) {
                Navigate( history.Pop(), addHistory: false );
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
                root.Invalidate();
            }
            Reload();
            Navigated?.Invoke( this, EventArgs.Empty );
        }

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

        private static List<AssetItem> List( string folder ) {
            try {
                DirectoryInfo dir = new DirectoryInfo( folder );
                IEnumerable<AssetItem> folders = dir.EnumerateDirectories( "*", listOptions ).OrderBy( d => d.Name, StringComparer.OrdinalIgnoreCase ).Select( d => new AssetItem( d ) );
                IEnumerable<AssetItem> files   = dir.EnumerateFiles( "*", listOptions ).OrderBy( f => f.Name, StringComparer.OrdinalIgnoreCase ).Select( f => new AssetItem( f ) );
                return folders.Concat( files ).ToList();
            } catch ( IOException ) {
                return new List<AssetItem>();
            } catch ( UnauthorizedAccessException ) {
                return new List<AssetItem>();
            }
        }

        private static List<AssetItem> Search( string folder, string query, CancellationToken cancel ) {
            List<AssetItem> results = new();
            foreach ( FileSystemInfo info in new DirectoryInfo( folder ).EnumerateFileSystemInfos( "*", searchOptions ) ) {
                cancel.ThrowIfCancellationRequested();
                if ( info.Name.Contains( query, StringComparison.OrdinalIgnoreCase ) ) {
                    string? location = Path.GetRelativePath( folder, Path.GetDirectoryName( info.FullName ) ?? folder ).Replace( '\\', '/' );
                    results.Add( new AssetItem( info ) { Location = location == "." ? null : location } );
                    if ( results.Count >= MaxSearchResults ) {
                        break;
                    }
                }
            }
            return results;
        }

        public async Task CookAsync( IReadOnlyList<AssetItem> items ) {
            // Folders expand to every cookable file underneath them.
            List<string> sources = new();
            foreach ( AssetItem item in items ) {
                if ( item.IsFolder ) {
                    sources.AddRange( Directory.EnumerateFiles( item.OsPath, "*", searchOptions ).Where( f => AssetKinds.IsCookable( AssetKinds.FromPath( f ) ) ) );
                } else if ( AssetKinds.IsCookable( item.Kind ) ) {
                    sources.Add( item.OsPath );
                }
            }
            if ( sources.Count == 0 ) {
                return;
            }

            IsBusy = true;
            int failed = 0;
            Stopwatch timer = Stopwatch.StartNew();
            Console.WriteLine( $"[AssetBrowser] Cooking {sources.Count} asset(s)..." );
            await Task.Run( () => {
                for ( int i = 0; i < sources.Count; i++ ) {
                    string source = sources[ i ];
                    string label = $"Cooking {i + 1}/{sources.Count}: {Path.GetFileName( source )}";
                    dispatcher.BeginInvoke( () => Status = label );

                    using Engine.Resource? cooker = AssetKinds.CreateCooker( AssetKinds.FromPath( source ) );
                    if ( cooker == null || !cooker.Cook( source, EditorPaths.ToCookedOsPath( source ) ) ) {
                        failed++;
                    }
                }
            } );
            IsBusy = false;

            foreach ( AssetItem item in Items ) {
                item.RefreshCookState();
            }
            string summary = $"Cooked {sources.Count - failed}/{sources.Count} in {timer.Elapsed.TotalSeconds:0.0}s" + ( failed > 0 ? $", {failed} failed (see Console)" : "" );
            Console.WriteLine( $"[AssetBrowser] {summary}" );
            UpdateStatus();
            Status = summary + "  ·  " + Status;
            SelectionChanged?.Invoke( this, EventArgs.Empty );   // so the inspector picks up the new cook state
        }

        private static void Open( AssetItem? item ) {
            if ( item != null ) {
                try {
                    Process.Start( new ProcessStartInfo( item.OsPath ) { UseShellExecute = true } );
                } catch ( Exception e ) {
                    Console.Error.WriteLine( $"[AssetBrowser] Couldn't open '{item.OsPath}': {e.Message}" );
                }
            }
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
                : $"{folders} folder(s), {files} file(s)";
            if ( Selection.Count == 1 ) {
                text += $"  ·  {Selection[ 0 ].ResourceName}";
            } else if ( Selection.Count > 1 ) {
                text += $"  ·  {Selection.Count} selected";
            }
            Status = text;
        }

        private bool IsRoot( string path ) => string.Equals( Path.GetFullPath( path ), RootPath, StringComparison.OrdinalIgnoreCase );
        private bool IsUnderRoot( string path ) => !Path.GetRelativePath( RootPath, path ).StartsWith( ".." );

        public void Dispose() {
            loadCancel.Cancel();
            watcher?.Dispose();
        }
    }
}
