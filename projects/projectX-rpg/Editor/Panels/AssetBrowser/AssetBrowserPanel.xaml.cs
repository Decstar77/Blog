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

        // ---- Dragging tiles (move into folders) ----

        // Drag payload for tiles dragged within the editor: string[] of OS paths. Kept separate from FileDrop so an
        // internal drag is a move, while files from Explorer are still imported (cooked).
        private const string MoveFormat = "Editor.AssetPaths";

        private Point dragStart;
        private AssetItem? pressedTile;
        private bool deferredSelect;

        private void AssetList_PreviewMouseLeftButtonDown( object sender, MouseButtonEventArgs e ) {
            pressedTile = TileUnder( e.OriginalSource );
            dragStart = e.GetPosition( AssetList );
            deferredSelect = false;
            // Pressing on a tile that's part of a multi-selection would normally collapse the selection to that tile,
            // which makes dragging several items impossible. Hold off until mouse-up, and only collapse if no drag started.
            if ( pressedTile != null && e.ClickCount == 1 && Keyboard.Modifiers == ModifierKeys.None
                 && AssetList.SelectedItems.Count > 1 && AssetList.SelectedItems.Contains( pressedTile ) ) {
                deferredSelect = true;
                AssetList.Focus();
                e.Handled = true;
            }
        }

        private void AssetList_PreviewMouseLeftButtonUp( object sender, MouseButtonEventArgs e ) {
            if ( deferredSelect && pressedTile != null ) {
                AssetList.SelectedItems.Clear();
                AssetList.SelectedItem = pressedTile;
            }
            deferredSelect = false;
            pressedTile = null;
        }

        private void AssetList_PreviewMouseMove( object sender, MouseEventArgs e ) {
            if ( pressedTile == null || e.LeftButton != MouseButtonState.Pressed ) {
                return;
            }
            Vector moved = e.GetPosition( AssetList ) - dragStart;
            if ( Math.Abs( moved.X ) < SystemParameters.MinimumHorizontalDragDistance && Math.Abs( moved.Y ) < SystemParameters.MinimumVerticalDragDistance ) {
                return;
            }

            List<AssetItem> dragged = AssetList.SelectedItems.Contains( pressedTile )
                ? AssetList.SelectedItems.Cast<AssetItem>().ToList()
                : new List<AssetItem> { pressedTile };
            pressedTile = null;
            deferredSelect = false;

            DataObject data = new DataObject( MoveFormat, dragged.Select( i => i.OsPath ).ToArray() );
            DragDrop.DoDragDrop( AssetList, data, DragDropEffects.Move );
            DropOverlay.Visibility = Visibility.Collapsed;
        }

        // ---- Drop: move (from tiles) or import (from Explorer) ----

        private void Browser_DragOver( object sender, DragEventArgs e ) {
            string? target = null;
            string verb = "";
            if ( e.Data.GetData( MoveFormat ) is string[] moving ) {
                target = DropTarget( e.OriginalSource );
                if ( target != null && !ViewModel.CanMove( moving, target ) ) {
                    target = null;
                }
                e.Effects = target != null ? DragDropEffects.Move : DragDropEffects.None;
                verb = "Move to";
            } else if ( e.Data.GetDataPresent( DataFormats.FileDrop ) ) {
                target = DropTarget( e.OriginalSource );
                e.Effects = target != null ? DragDropEffects.Copy : DragDropEffects.None;
                verb = "Add to";
            } else {
                e.Effects = DragDropEffects.None;
            }
            e.Handled = true;
            if ( target != null ) {
                DropLabel.Text = $"{verb} {ViewModel.DisplayPath( target )}";
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
            if ( DropTarget( e.OriginalSource ) is not { } target ) {
                return;
            }
            if ( e.Data.GetData( MoveFormat ) is string[] moving ) {
                e.Handled = true;
                Dispatcher.BeginInvoke( () => ViewModel.Move( moving, target ) );
            } else if ( e.Data.GetData( DataFormats.FileDrop ) is string[] paths ) {
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
