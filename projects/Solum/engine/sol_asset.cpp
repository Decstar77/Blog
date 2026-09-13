#include "sol_asset.h"

#include <cstdio>
#include <cstring>

namespace sol {

    // fopen/fwrite need a null-terminated buffer; StringView is not one. Paths
    // in this engine are short, so a stack copy is fine.
    static bool CopyToCString( StringView str, char * buffer, i32 bufSize ) {
        if( str.count < 0 || str.count >= bufSize ) {
            return false;
        }

        memcpy( buffer, str.data, (size_t)str.count );
        buffer[str.count] = '\0';
        return true;
    }

    // Everything up to (not including) the last slash. Empty if path has none.
    static StringView PathDirectory( StringView path ) {
        const i32 slash = StringFindLastChar( path, '/' );
        const i32 backslash = StringFindLastChar( path, '\\' );
        const i32 cut = slash > backslash ? slash : backslash;
        if( cut < 0 ) {
            return StringView( "", 0 );
        }
        return StringView( path.data, cut );
    }

    // ---- file helpers ------------------------------------------------------

    bool FileReadEntire( StringView path, List<u8> & outBytes ) {
        char pathBuf[1024];
        if( CopyToCString( path, pathBuf, sizeof( pathBuf ) ) == false ) {
            fprintf( stderr, "FileReadEntire: path too long\n" );
            return false;
        }

        FILE * file = fopen( pathBuf, "rb" );
        if( file == nullptr ) {
            fprintf( stderr, "FileReadEntire: failed to open '%s'\n", pathBuf );
            return false;
        }

        if( fseek( file, 0, SEEK_END ) != 0 ) {
            fprintf( stderr, "FileReadEntire: seek failed on '%s'\n", pathBuf );
            fclose( file );
            return false;
        }

        const long size = ftell( file );
        if( size < 0 ) {
            fprintf( stderr, "FileReadEntire: tell failed on '%s'\n", pathBuf );
            fclose( file );
            return false;
        }

        if( fseek( file, 0, SEEK_SET ) != 0 ) {
            fprintf( stderr, "FileReadEntire: rewind failed on '%s'\n", pathBuf );
            fclose( file );
            return false;
        }

        List<u8> bytes = {};
        ListResize( bytes, (i32)size );
        if( bytes.count != (i32)size ) {
            fprintf( stderr, "FileReadEntire: allocation failed for '%s'\n", pathBuf );
            fclose( file );
            ListFree( bytes );
            return false;
        }

        if( size > 0 ) {
            const size_t readCount = fread( bytes.data, 1, (size_t)size, file );
            if( readCount != (size_t)size ) {
                fprintf( stderr, "FileReadEntire: short read on '%s'\n", pathBuf );
                fclose( file );
                ListFree( bytes );
                return false;
            }
        }

        fclose( file );
        outBytes = bytes;
        return true;
    }

    bool FileWriteEntire( StringView path, const void * data, i32 byteCount ) {
        char pathBuf[1024];
        if( CopyToCString( path, pathBuf, sizeof( pathBuf ) ) == false ) {
            fprintf( stderr, "FileWriteEntire: path too long\n" );
            return false;
        }

        if( byteCount < 0 || ( byteCount > 0 && data == nullptr ) ) {
            fprintf( stderr, "FileWriteEntire: invalid buffer for '%s'\n", pathBuf );
            return false;
        }

        FILE * file = fopen( pathBuf, "wb" );
        if( file == nullptr ) {
            fprintf( stderr, "FileWriteEntire: failed to open '%s'\n", pathBuf );
            return false;
        }

        if( byteCount > 0 ) {
            const size_t written = fwrite( data, 1, (size_t)byteCount, file );
            if( written != (size_t)byteCount ) {
                fprintf( stderr, "FileWriteEntire: short write on '%s'\n", pathBuf );
                fclose( file );
                return false;
            }
        }

        fclose( file );
        return true;
    }

    // ---- sidecar text --------------------------------------------------------

    TextureMeta TextureMetaDefault() {
        TextureMeta meta = {};
        meta.format = TextureFormat_RGBA8_SRGB;
        meta.filter = TextureFilter_Linear;
        meta.wrap = TextureWrap_Repeat;
        meta.generateMips = false;
        return meta;
    }

    bool TextureMetaParse( StringView text, TextureMeta * outMeta ) {
        TextureMeta meta = TextureMetaDefault();

        StringView cursor = text;
        StringView line = {};
        while( StringSplitNext( &cursor, '\n', &line ) ) {
            const StringView trimmed = StringTrim( line );
            if( trimmed.count == 0 || trimmed.data[0] == '#' ) {
                continue;
            }

            const i32 eq = StringFindChar( trimmed, '=' );
            if( eq < 0 ) {
                // Malformed line, no key/value to be found - ignore rather than fail.
                continue;
            }

            const StringView key = StringTrim( StringSubstring( trimmed, 0, eq ) );
            const StringView value = StringTrim( StringSubstring( trimmed, eq + 1, trimmed.count - eq - 1 ) );

            if( StringEqualsIgnoreCase( key, "source" ) ) {
                if( StringSet( meta.source, value ) == false ) {
                    fprintf( stderr, "TextureMetaParse: source value too long\n" );
                    return false;
                }
            } else if( StringEqualsIgnoreCase( key, "binary" ) ) {
                if( StringSet( meta.binary, value ) == false ) {
                    fprintf( stderr, "TextureMetaParse: binary value too long\n" );
                    return false;
                }
            } else if( StringEqualsIgnoreCase( key, "format" ) ) {
                if( StringEqualsIgnoreCase( value, "srgb" ) ) {
                    meta.format = TextureFormat_RGBA8_SRGB;
                } else if( StringEqualsIgnoreCase( value, "unorm" ) ) {
                    meta.format = TextureFormat_RGBA8_UNORM;
                } else {
                    fprintf( stderr, "TextureMetaParse: bad format '%.*s'\n", value.count, value.data );
                    return false;
                }
            } else if( StringEqualsIgnoreCase( key, "filter" ) ) {
                if( StringEqualsIgnoreCase( value, "nearest" ) ) {
                    meta.filter = TextureFilter_Nearest;
                } else if( StringEqualsIgnoreCase( value, "linear" ) ) {
                    meta.filter = TextureFilter_Linear;
                } else {
                    fprintf( stderr, "TextureMetaParse: bad filter '%.*s'\n", value.count, value.data );
                    return false;
                }
            } else if( StringEqualsIgnoreCase( key, "wrap" ) ) {
                if( StringEqualsIgnoreCase( value, "repeat" ) ) {
                    meta.wrap = TextureWrap_Repeat;
                } else if( StringEqualsIgnoreCase( value, "clamp" ) ) {
                    meta.wrap = TextureWrap_Clamp;
                } else if( StringEqualsIgnoreCase( value, "mirror" ) ) {
                    meta.wrap = TextureWrap_Mirror;
                } else {
                    fprintf( stderr, "TextureMetaParse: bad wrap '%.*s'\n", value.count, value.data );
                    return false;
                }
            } else if( StringEqualsIgnoreCase( key, "mips" ) ) {
                if( StringEqualsIgnoreCase( value, "true" ) ) {
                    meta.generateMips = true;
                } else if( StringEqualsIgnoreCase( value, "false" ) ) {
                    meta.generateMips = false;
                } else {
                    fprintf( stderr, "TextureMetaParse: bad mips '%.*s'\n", value.count, value.data );
                    return false;
                }
            }
            // Unknown keys (including "version"/"type") are ignored on purpose -
            // this is what lets an older engine load a newer sidecar.
        }

        *outMeta = meta;
        return true;
    }

    bool TextureMetaWrite( const TextureMeta & meta, HeapString & outText ) {
        HeapStringClear( outText );

        StringView formatStr = meta.format == TextureFormat_RGBA8_UNORM ? StringView( "unorm" ) : StringView( "srgb" );
        StringView filterStr = meta.filter == TextureFilter_Nearest ? StringView( "nearest" ) : StringView( "linear" );
        StringView wrapStr = "repeat";
        if( meta.wrap == TextureWrap_Clamp ) {
            wrapStr = "clamp";
        } else if( meta.wrap == TextureWrap_Mirror ) {
            wrapStr = "mirror";
        }
        StringView mipsStr = meta.generateMips ? StringView( "true" ) : StringView( "false" );

        bool ok = true;
        ok = StringAppend( outText, "# Solum texture asset\n" ) && ok;
        ok = StringAppend( outText, "version = 1\n" ) && ok;
        ok = StringAppend( outText, "type = texture\n" ) && ok;
        ok = StringAppend( outText, "source = " ) && ok;
        ok = StringAppend( outText, meta.source ) && ok;
        ok = StringAppend( outText, "\n" ) && ok;
        ok = StringAppend( outText, "binary = " ) && ok;
        ok = StringAppend( outText, meta.binary ) && ok;
        ok = StringAppend( outText, "\n" ) && ok;
        ok = StringAppend( outText, "format = " ) && ok;
        ok = StringAppend( outText, formatStr ) && ok;
        ok = StringAppend( outText, "\n" ) && ok;
        ok = StringAppend( outText, "filter = " ) && ok;
        ok = StringAppend( outText, filterStr ) && ok;
        ok = StringAppend( outText, "\n" ) && ok;
        ok = StringAppend( outText, "wrap = " ) && ok;
        ok = StringAppend( outText, wrapStr ) && ok;
        ok = StringAppend( outText, "\n" ) && ok;
        ok = StringAppend( outText, "mips = " ) && ok;
        ok = StringAppend( outText, mipsStr ) && ok;
        ok = StringAppend( outText, "\n" ) && ok;

        return ok;
    }

    // ---- whole assets --------------------------------------------------------

    bool TextureAssetLoad( StringView metaPath, TextureAsset * outAsset ) {
        List<u8> metaBytes = {};
        if( FileReadEntire( metaPath, metaBytes ) == false ) {
            fprintf( stderr, "TextureAssetLoad: failed to read meta '%.*s'\n", metaPath.count, metaPath.data );
            return false;
        }

        TextureMeta meta = {};
        const bool parsed = TextureMetaParse( StringView( (const char *)metaBytes.data, metaBytes.count ), &meta );
        ListFree( metaBytes );
        if( parsed == false ) {
            fprintf( stderr, "TextureAssetLoad: failed to parse meta '%.*s'\n", metaPath.count, metaPath.data );
            return false;
        }

        const StringView dir = PathDirectory( metaPath );
        FixedString<1024> binPath = {};
        if( dir.count > 0 ) {
            StringAppend( binPath, dir );
            StringAppend( binPath, "/" );
        }
        StringAppend( binPath, meta.binary );

        List<u8> binBytes = {};
        if( FileReadEntire( binPath, binBytes ) == false ) {
            fprintf( stderr, "TextureAssetLoad: failed to read payload '%.*s'\n", binPath.count, binPath.data );
            return false;
        }

        if( binBytes.count < (i32)sizeof( TextureBinHeader ) ) {
            fprintf( stderr, "TextureAssetLoad: payload smaller than header '%.*s'\n", binPath.count, binPath.data );
            ListFree( binBytes );
            return false;
        }

        TextureBinHeader header = {};
        memcpy( &header, binBytes.data, sizeof( header ) );

        if( header.magic != kTextureMagic ) {
            fprintf( stderr, "TextureAssetLoad: bad magic in '%.*s'\n", binPath.count, binPath.data );
            ListFree( binBytes );
            return false;
        }
        if( header.version != kTextureVersion ) {
            fprintf( stderr, "TextureAssetLoad: unsupported version in '%.*s'\n", binPath.count, binPath.data );
            ListFree( binBytes );
            return false;
        }
        if( header.width <= 0 || header.height <= 0 ) {
            fprintf( stderr, "TextureAssetLoad: bad dimensions in '%.*s'\n", binPath.count, binPath.data );
            ListFree( binBytes );
            return false;
        }
        if( header.channels != 4 ) {
            fprintf( stderr, "TextureAssetLoad: unsupported channel count in '%.*s'\n", binPath.count, binPath.data );
            ListFree( binBytes );
            return false;
        }

        const u32 expectedBytes = (u32)header.width * (u32)header.height * 4u;
        const i32 remaining = binBytes.count - (i32)sizeof( header );
        if( header.dataBytes != expectedBytes || (i32)header.dataBytes != remaining ) {
            fprintf( stderr, "TextureAssetLoad: payload size mismatch in '%.*s'\n", binPath.count, binPath.data );
            ListFree( binBytes );
            return false;
        }

        List<u8> pixels = {};
        ListResize( pixels, (i32)header.dataBytes );
        if( pixels.count != (i32)header.dataBytes ) {
            fprintf( stderr, "TextureAssetLoad: allocation failed for '%.*s'\n", binPath.count, binPath.data );
            ListFree( binBytes );
            ListFree( pixels );
            return false;
        }
        if( header.dataBytes > 0 ) {
            memcpy( pixels.data, binBytes.data + sizeof( header ), header.dataBytes );
        }
        ListFree( binBytes );

        TextureAsset asset = {};
        asset.meta = meta;
        asset.width = header.width;
        asset.height = header.height;
        asset.pixels = pixels;

        *outAsset = asset;
        return true;
    }

    void TextureAssetFree( TextureAsset * asset ) {
        ListFree( asset->pixels );
        *asset = TextureAsset{};
    }

    bool TextureAssetWrite( StringView outputDirectory, StringView assetName,
                            const TextureMeta & meta,
                            const void * pixels, i32 width, i32 height ) {
        if( width <= 0 || height <= 0 || pixels == nullptr ) {
            fprintf( stderr, "TextureAssetWrite: invalid image data\n" );
            return false;
        }

        TextureMeta metaCopy = meta;
        FixedString<256> binaryName = {};
        StringAppend( binaryName, assetName );
        StringAppend( binaryName, ".stex" );
        if( StringSet( metaCopy.binary, binaryName ) == false ) {
            fprintf( stderr, "TextureAssetWrite: asset name too long\n" );
            return false;
        }

        TextureBinHeader header = {};
        header.magic = kTextureMagic;
        header.version = kTextureVersion;
        header.width = width;
        header.height = height;
        header.channels = 4;
        header.format = (u32)metaCopy.format;
        header.mipCount = 1;
        header.dataBytes = (u32)width * (u32)height * 4u;

        List<u8> payload = {};
        ListResize( payload, (i32)sizeof( header ) + (i32)header.dataBytes );
        if( payload.count != (i32)sizeof( header ) + (i32)header.dataBytes ) {
            fprintf( stderr, "TextureAssetWrite: allocation failed\n" );
            ListFree( payload );
            return false;
        }
        memcpy( payload.data, &header, sizeof( header ) );
        memcpy( payload.data + sizeof( header ), pixels, header.dataBytes );

        FixedString<1024> stexPath = {};
        StringAppend( stexPath, outputDirectory );
        StringAppend( stexPath, "/" );
        StringAppend( stexPath, assetName );
        StringAppend( stexPath, ".stex" );

        const bool binOk = FileWriteEntire( stexPath, payload.data, payload.count );
        ListFree( payload );
        if( binOk == false ) {
            fprintf( stderr, "TextureAssetWrite: failed to write '%.*s'\n", stexPath.count, stexPath.data );
            return false;
        }

        HeapString metaText = {};
        const bool metaBuilt = TextureMetaWrite( metaCopy, metaText );
        if( metaBuilt == false ) {
            fprintf( stderr, "TextureAssetWrite: failed to serialise meta for '%.*s'\n", assetName.count, assetName.data );
            HeapStringFree( metaText );
            return false;
        }

        FixedString<1024> metaPath = {};
        StringAppend( metaPath, outputDirectory );
        StringAppend( metaPath, "/" );
        StringAppend( metaPath, assetName );
        StringAppend( metaPath, ".meta" );

        const bool metaOk = FileWriteEntire( metaPath, metaText.data, metaText.count );
        HeapStringFree( metaText );
        if( metaOk == false ) {
            fprintf( stderr, "TextureAssetWrite: failed to write '%.*s'\n", metaPath.count, metaPath.data );
            return false;
        }

        return true;
    }

} // namespace sol
