using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

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
            if ( e.ChangedButton == MouseButton.Left
                 && ItemsControl.ContainerFromElement( AssetList, (DependencyObject)e.OriginalSource ) is ListBoxItem { DataContext: AssetItem item } ) {
                ViewModel.Activate( item );
            }
        }

        private void AssetList_KeyDown( object sender, KeyEventArgs e ) {
            switch ( e.Key ) {
                case Key.Enter when AssetList.SelectedItem is AssetItem item:
                    if ( item.IsFolder ) {
                        ViewModel.Navigate( item.OsPath );
                    } else {
                        ViewModel.CookCommand.Execute( null );
                    }
                    e.Handled = true;
                    break;
                case Key.Back:
                    ViewModel.UpCommand.Execute( null );
                    e.Handled = true;
                    break;
            }
        }
    }
}
