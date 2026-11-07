namespace Engine {
    public abstract class Resource : IDisposable {
        public string           Name => File?.FileName ?? "";
        public ResourceFile?    File { get; private set; }

        public void Initialize( ResourceFile file ) {
            File = file;
        }

        // Turn a source asset (png, wav, ...) into the engine format at cookedOsPath. Return false on failure.
        public abstract bool Cook( string assetOsPath, string cookedOsPath );

        // Build the runtime object (GPU texture etc.) from a cooked file. Return false on failure.
        public abstract bool LoadFromSource( ResourceFile resourceFile );

        public virtual void Dispose() { }
    }
}
