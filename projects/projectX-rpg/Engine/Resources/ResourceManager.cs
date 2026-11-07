namespace Engine.Resources {
    // Owns every loaded resource. Construct once after the window (it needs a GL context) and dispose
    // before it, e.g. `using ResourceManager resources = new ResourceManager( ... );` right after the window.
    //
    // Lookup order for LoadResource( "Textures/grass.png" ):
    //   1. <basePath>/Cooked/Textures/grass.png.res   (engine format, loaded directly)
    //   2. <assetPath>/Textures/grass.png             (source asset, cooked to 1. then loaded)
    public sealed class ResourceManager : IDisposable {
        private static ResourceManager? instance;
        public static ResourceManager Instance => instance ?? throw new InvalidOperationException( "ResourceManager has not been created" );

        public const string CookedFolder = "Cooked";
        public const string CookedExtension = ".res";

        public string BasePath { get; }
        public string AssetPath { get; }
        public string CookedPath => Path.Combine( BasePath, CookedFolder );

        private readonly Dictionary<string, Resource> resources = new();

        public ResourceManager( string basePath, string assetPath ) {
            if ( instance != null ) {
                throw new InvalidOperationException( "ResourceManager already exists" );
            }
            BasePath = basePath;
            AssetPath = assetPath;
            instance = this;
        }

        public static T? Load<T>( string resourceName ) where T : Resource, new() => Instance.LoadResource<T>( resourceName );

        // Returns null if the resource has neither a cooked file nor a source asset, or fails to load.
        public T? LoadResource<T>( string resourceName ) where T : Resource, new() {
            resourceName = NormalizeName( resourceName );

            if ( resources.TryGetValue( resourceName, out Resource? existing ) ) {
                if ( existing is T typed ) {
                    return typed;
                }
                throw new InvalidOperationException( $"Resource '{resourceName}' is loaded as {existing.GetType().Name}, not {typeof( T ).Name}" );
            }

            T resource = new T();

            string cookedOsPath = Path.Combine( CookedPath, resourceName + CookedExtension );
            ResourceFile file = new ResourceFile( resourceName, cookedOsPath );
            if ( !file.Exists ) {
                string assetOsPath = Path.Combine( AssetPath, resourceName );
                if ( !File.Exists( assetOsPath ) || !resource.Cook( assetOsPath, cookedOsPath ) ) {
                    Console.Error.WriteLine( $"[ResourceManager] Could not find or cook '{resourceName}'" );
                    return null;
                }
                file = new ResourceFile( resourceName, cookedOsPath );
                if ( !file.Exists ) {
                    return null;
                }
            }

            resource.Initialize( file );
            if ( !resource.LoadFromSource( file ) ) {
                Console.Error.WriteLine( $"[ResourceManager] Failed to load '{resourceName}'" );
                resource.Dispose();
                return null;
            }

            resources[ resourceName ] = resource;
            return resource;
        }

        public void Unload( string resourceName ) {
            resourceName = NormalizeName( resourceName );
            if ( resources.Remove( resourceName, out Resource? resource ) ) {
                resource.Dispose();
            }
        }

        public void Dispose() {
            foreach ( Resource resource in resources.Values ) {
                resource.Dispose();
            }
            resources.Clear();
            if ( instance == this ) {
                instance = null;
            }
        }

        // So "Textures\a.png" and "textures/a.png" share one cache entry.
        private static string NormalizeName( string name ) => name.Replace( '\\', '/' ).TrimStart( '/' ).ToLowerInvariant();
    }
}
