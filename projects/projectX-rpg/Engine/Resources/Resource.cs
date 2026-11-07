namespace Engine {
    public abstract class Resource {
        private ResourceFile? resourceFile = null;

        public void Initialize(ResourceFile file) {
            resourceFile = file;
        }

        public abstract bool LoadFromSource( ResourceFile resourceFile );
    }
}
