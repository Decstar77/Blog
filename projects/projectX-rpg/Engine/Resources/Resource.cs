namespace Engine {
    public abstract class Resource {
        public string   Name { get; protected set; } = "";

        public void Initialize(string resourceName) {
            Name = resourceName;
        }

        public abstract bool LoadFromSource( ResourceFile resourceFile );
    }
}
