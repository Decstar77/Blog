using System.Runtime.InteropServices;

namespace Engine {
    // A cooked file read fully into memory, with a read cursor for pulling out headers and payloads.
    public class ResourceFile {
        public string   FileName        { get; set; } = "";
        public string   OsPath          { get; set; } = "";
        public byte[]   Data            { get; private set; } = new byte[0];
        public int      Position        { get; set; }

        public bool     Exists => Data.Length != 0;
        public int      Remaining => Data.Length - Position;

        public ResourceFile( string fileName, string osPath ) {
            FileName = fileName;
            OsPath = osPath;
            if ( System.IO.File.Exists( osPath ) ) {
                Data = System.IO.File.ReadAllBytes( osPath );
            }
        }

        public T ReadStruct<T>() where T : unmanaged {
            int size = Marshal.SizeOf<T>();
            if ( Remaining < size ) {
                throw new EndOfStreamException( $"{FileName}: need {size} bytes for {typeof( T ).Name}, {Remaining} left" );
            }
            T value = MemoryMarshal.Read<T>( Data.AsSpan( Position, size ) );
            Position += size;
            return value;
        }

        public ReadOnlySpan<byte> ReadBytes( int count ) {
            if ( Remaining < count ) {
                throw new EndOfStreamException( $"{FileName}: need {count} bytes, {Remaining} left" );
            }
            ReadOnlySpan<byte> span = Data.AsSpan( Position, count );
            Position += count;
            return span;
        }

        // Writes header followed by payload, creating the directory if needed. Used by the cookers.
        public static void Write<T>( string osPath, in T header, ReadOnlySpan<byte> payload ) where T : unmanaged {
            string? dir = Path.GetDirectoryName( osPath );
            if ( !string.IsNullOrEmpty( dir ) ) {
                Directory.CreateDirectory( dir );
            }
            using FileStream stream = System.IO.File.Create( osPath );
            T copy = header;
            stream.Write( MemoryMarshal.AsBytes( new ReadOnlySpan<T>( ref copy ) ) );
            stream.Write( payload );
        }
    }
}
