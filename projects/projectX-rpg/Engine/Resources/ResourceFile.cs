namespace Engine {
    public class ResourceFile {
        public string   FileName        { get; set; } = "";
        public string   OsPath          { get; set; } = "";
        public byte[]   Data            { get; private set; } = new byte[0];

        public bool     Exists => Data.Length != 0;

        public ResourceFile( string fileName, string osPath ) {
            FileName = fileName;
            OsPath = osPath;
            if ( File.Exists( osPath ) ) {
                Data = File.ReadAllBytes( osPath );
            }
        }
    }
}
