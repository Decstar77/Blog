using System;
using System.Collections.Generic;
using System.Text;

namespace Engine.Resources {
    public static class ResourceManager {
        private static string                           basePath = "";
        private static string                           assetPath = "";

        private static Dictionary<string, Resource>     resources = new Dictionary<string, Resource>();

        public static void Intialize( string basePath_, string assetPath_ ) {
            basePath = basePath_;
            assetPath = assetPath_;
        }

        public static T LoadResource<T>( string resourceName ) where T : Resource, new() {
            if ( resources.TryGetValue( resourceName, out Resource? existingResource ) ) {
                return (T)existingResource;
            }

            Resource resource = new T();

            string osPath = Path.Combine(basePath, resourceName);
            ResourceFile file = new ResourceFile(resourceName, osPath);
            if ( file.Exists == true ) {
                resource.Initialize( file );
                resource.LoadFromSource( file );
                resources[resourceName] = resource;
            } else {
                string assetOsPath = Path.Combine(assetPath, resourceName);
                file = ImageCooker.Cook( assetOsPath );
                if ( file.Exists ) {
                    resource.Initialize( file );
                    resource.LoadFromSource( file );
                    resources[resourceName] = resource;
                }
            }

            return (T)resource;
        }

    }
}
