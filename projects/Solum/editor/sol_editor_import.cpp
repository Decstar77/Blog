#include "sol_editor_import.h"

#include <cstdio>
#include <cstring>

#include <stb_image.h>

namespace sol {

    // stbi_load needs a null-terminated buffer; StringView is not one. Import
    // paths come from the command line and are short, so a stack copy is fine.
    static bool CopyToCString( StringView str, char * buffer, i32 bufSize ) {
        if( str.count < 0 || str.count >= bufSize ) {
            return false;
        }

        memcpy( buffer, str.data, (size_t)str.count );
        buffer[str.count] = '\0';
        return true;
    }

    bool ImportTexture( StringView sourcePath, StringView outputDirectory, StringView assetName,
                        TextureFormat format, TextureFilter filter, TextureWrap wrap ) {
        char sourcePathBuf[1024];
        if( CopyToCString( sourcePath, sourcePathBuf, sizeof( sourcePathBuf ) ) == false ) {
            fprintf( stderr, "ImportTexture: source path too long\n" );
            return false;
        }

        i32 width = 0;
        i32 height = 0;
        i32 sourceChannels = 0;
        // req_comp = 4 forces RGBA output even when the source has no alpha
        // channel (e.g. PNG colour type 2) - the engine contract requires
        // channels == 4 in the payload, so alpha comes back as 255 in that case.
        u8 * pixels = stbi_load( sourcePathBuf, &width, &height, &sourceChannels, 4 );
        if( pixels == nullptr ) {
            fprintf( stderr, "ImportTexture: failed to decode '%s': %s\n", sourcePathBuf, stbi_failure_reason() );
            return false;
        }

        TextureMeta meta = TextureMetaDefault();
        meta.format = format;
        meta.filter = filter;
        meta.wrap = wrap;
        if( StringSet( meta.source, sourcePath ) == false ) {
            fprintf( stderr, "ImportTexture: source path too long for meta\n" );
            stbi_image_free( pixels );
            return false;
        }

        const bool written = TextureAssetWrite( outputDirectory, assetName, meta, pixels, width, height );
        stbi_image_free( pixels );

        if( written == false ) {
            fprintf( stderr, "ImportTexture: failed to write asset '%.*s'\n", assetName.count, assetName.data );
            return false;
        }

        return true;
    }

} // namespace sol
