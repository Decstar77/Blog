using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace Editor.Panels.AssetBrowser {
    public partial class AssetBrowserPanel : UserControl {
        public AssetBrowserViewModel ViewModel { get; }

        public AssetBrowserPanel( AssetBrowserViewModel viewModel ) {
            InitializeComponent();
            ViewModel = viewModel;
            DataContext = viewModel;
            viewModel.Navigated += ( _, _ ) => SyncTree();

            // Ctrl+F from anywhere in the panel jumps to the search box.
            InputBindings.Add( new KeyBinding( new Core.RelayCommand( () => { SearchBox.Focus(); SearchBox.SelectAll(); } ), Key.F, ModifierKeys.Control ) );
            // Mouse back button.
            PreviewMouseDown += ( _, e ) => {
                if ( e.ChangedButton == MouseButton.XButton1 && ViewModel.BackCommand.CanExecute( null ) ) {
                    ViewModel.BackCommand.Execute( null );
                    e.Handled = true;
                }
            };
            Loaded += ( _, _ ) => SyncTree();
        }

        private void SyncTree() {
            FolderNode? node = ViewModel.Roots.FirstOrDefault()?.Reveal( ViewModel.CurrentPath );
            if ( node != null ) {
                node.IsSelected = true;
            }
        }

        private void FolderTree_SelectedItemChanged( object sender, RoutedPropertyChangedEventArgs<object> e ) {
            if ( e.NewValue is FolderNode node ) {
                ViewModel.Navigate( node.OsPath );
            }
        }

        private void AssetList_SelectionChanged( object sender, SelectionChangedEventArgs e ) {
            ViewModel.Selection = AssetList.SelectedItems.Cast<AssetItem>().ToList();
        }

        private void AssetList_MouseDoubleClick( object sender, MouseButtonEventArgs e ) {
            // Only react to double clicks on a tile, not on empty space or the scrollbar.
            if ( e.ChangedButton == MouseButton.Left && TileUnder( e.OriginalSource ) is { } item ) {
                ViewModel.Activate( item );
            }
        }

        private void AssetList_KeyDown( object sender, KeyEventArgs e ) {
            switch ( e.Key ) {
                case Key.Enter when AssetList.SelectedItem is AssetItem item:
                    ViewModel.Activate( item );
                    e.Handled = true;
                    break;
                case Key.Back:
                    ViewModel.UpCommand.Execute( null );
                    e.Handled = true;
                    break;
                case Key.Delete when ViewModel.DeleteCommand.CanExecute( null ):
                    ViewModel.DeleteCommand.Execute( null );
                    e.Handled = true;
                    break;
            }
        }

        // ---- Drag and drop import ----

        private void Browser_DragOver( object sender, DragEventArgs e ) {
            string? target = e.Data.GetDataPresent( DataFormats.FileDrop ) ? DropTarget( e.OriginalSource ) : null;
            e.Effects = target != null ? DragDropEffects.Copy : DragDropEffects.None;
            e.Handled = true;
            if ( target != null ) {
                DropLabel.Text = $"Add to {ViewModel.DisplayPath( target )}";
                DropOverlay.Visibility = Visibility.Visible;
            } else {
                DropOverlay.Visibility = Visibility.Collapsed;
            }
        }

        private void Browser_DragLeave( object sender, DragEventArgs e ) {
            DropOverlay.Visibility = Visibility.Collapsed;
        }

        private void Browser_Drop( object sender, DragEventArgs e ) {
            DropOverlay.Visibility = Visibility.Collapsed;
            if ( e.Data.GetData( DataFormats.FileDrop ) is string[] paths && DropTarget( e.OriginalSource ) is { } target ) {
                e.Handled = true;
                // Let the drag source (Explorer) finish before we potentially pop up a dialog.
                Dispatcher.BeginInvoke( () => _ = ViewModel.ImportAsync( paths, target ) );
            }
        }

        // Dropping onto a folder tile or a tree node targets that folder; anywhere else targets the current folder.
        private string? DropTarget( object source ) {
            if ( ViewModel.IsBusy ) {
                return null;
            }
            if ( TileUnder( source ) is { IsFolder: true } folder ) {
                return folder.OsPath;
            }
            if ( FindAncestor<TreeViewItem>( source as DependencyObject ) is { DataContext: FolderNode node } ) {
                return node.OsPath;
            }
            return ViewModel.CurrentPath;
        }

        private AssetItem? TileUnder( object source ) =>
            FindAncestor<ListBoxItem>( source as DependencyObject )?.DataContext as AssetItem;

        private static T? FindAncestor<T>( DependencyObject? d ) where T : DependencyObject {
            while ( d != null && d is not T ) {
                d = d is Visual or System.Windows.Media.Media3D.Visual3D ? VisualTreeHelper.GetParent( d ) : LogicalTreeHelper.GetParent( d );
            }
            return d as T;
        }
    }
}
