namespace Engine.Resources {
    public sealed class ResourceManager : IDisposable {
        private static ResourceManager? instance;
        public static ResourceManager Instance => instance ?? throw new InvalidOperationException( "ResourceManager has not been created" );

        // Everything the engine loads lives under <BasePath>/Resources as cooked "<resource name>.res" files.
        // Raw source assets (png, wav, ...) are cooked into there by the editor; the engine never sees them.
        public const string ResourcesFolder = "Resources";
        public const string Extension = ".res";

        public string BasePath { get; }
        public string ResourcesPath => Path.Combine( BasePath, ResourcesFolder );

        private readonly Dictionary<string, Resource> resources = new();

        public ResourceManager( string basePath ) {
            if ( instance != null ) {
                throw new InvalidOperationException( "ResourceManager already exists" );
            }
            BasePath = basePath;
            instance = this;
        }

        public static T? Load<T>( string resourceName ) where T : Resource, new() => Instance.LoadResource<T>( resourceName );

        // Returns null if the resource doesn't exist or fails to load.
        public T? LoadResource<T>( string resourceName ) where T : Resource, new() {
            resourceName = NormalizeName( resourceName );

            if ( resources.TryGetValue( resourceName, out Resource? existing ) ) {
                if ( existing is T typed ) {
                    return typed;
                }
                throw new InvalidOperationException( $"Resource '{resourceName}' is loaded as {existing.GetType().Name}, not {typeof( T ).Name}" );
            }

            T resource = new T();

            ResourceFile file = new ResourceFile( resourceName, ToOsPath( resourceName ) );
            if ( !file.Exists ) {
                Console.Error.WriteLine( $"[ResourceManager] Resource '{resourceName}' not found at {file.OsPath}" );
                return null;
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

        public string ToOsPath( string resourceName ) => Path.Combine( ResourcesPath, NormalizeName( resourceName ) + Extension );

        // So "Textures\a.png" and "textures/a.png" share one cache entry.
        private static string NormalizeName( string name ) => name.Replace( '\\', '/' ).TrimStart( '/' ).ToLowerInvariant();
    }
}
