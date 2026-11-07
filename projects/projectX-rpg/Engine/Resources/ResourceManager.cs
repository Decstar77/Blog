using System;
using System.Collections.Generic;
using System.Text;

namespace Engine.Resources {
    public static class ResourceManager {
        private static string                           basePath = "";
        private static Dictionary<string, Resource>     resources = new Dictionary<string, Resource>();

        public static void Intialize( string basePath_ ) {
            basePath = basePath_;
        }

        public static T LoadResource<T>( string resourceName ) where T : Resource, new() {
            if ( resources.TryGetValue( resourceName, out Resource? existingResource ) ) {
                return (T)existingResource;
            }

            string osPath = Path.Combine(basePath, resourceName);
            ResourceFile file = new ResourceFile(resourceName, osPath);

            Resource resource = new T();
            resource.Initialize( resourceName );
            resource.LoadFromSource( file );
            resources[resourceName] = resource;
            return (T)resource;
        }

    }
}
