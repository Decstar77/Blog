#pragma once
#include "sol_defines.h"
#include "sol_list.h"
#include "sol_string.h"

// The engine half of the asset pipeline. Nothing in here knows what a PNG is:
// the editor decodes source art and writes the two files described below, and
// the engine only ever reads them back.
//
// Every asset is a pair:
//   <name>.stex  - binary payload, fully decompressed, ready to memcpy to the GPU
//   <name>.meta  - text sidecar: where it came from and how to sample it
//
// The sidecar is text on purpose. It diffs, it merges, and it can be edited by
// hand without a tool. The payload is binary on purpose: the engine should do a
// header read and one bulk copy, nothing more.

namespace sol {

    // ---- file helpers ------------------------------------------------------

    bool FileReadEntire( StringView path, List<u8> & outBytes );
    bool FileWriteEntire( StringView path, const void * data, i32 byteCount );

    // ---- texture assets ----------------------------------------------------

    enum TextureFormat : u32 {
        TextureFormat_RGBA8_UNORM = 0,   // data, normals, masks
        TextureFormat_RGBA8_SRGB  = 1,   // colour art, the usual case
    };

    enum TextureFilter : u32 {
        TextureFilter_Nearest = 0,
        TextureFilter_Linear  = 1,
    };

    enum TextureWrap : u32 {
        TextureWrap_Repeat = 0,
        TextureWrap_Clamp  = 1,
        TextureWrap_Mirror = 2,
    };

    // Mirrors the .meta sidecar one field per line. Unknown keys are ignored on
    // read so an older engine still loads a newer sidecar.
    struct TextureMeta {
        LargeString     source;         // absolute path to the original art
        LargeString     binary;         // payload filename, relative to the .meta
        TextureFormat   format;
        TextureFilter   filter;
        TextureWrap     wrap;
        bool            generateMips;   // honoured by the importer, not the engine
    };

    // Byte layout of the .stex payload. Written and read as a flat struct, so it
    // must stay 4-byte-field, little-endian, and append-only across versions.
    //
    //   magic 'SOLT' | version | width | height | channels | format | mipCount | dataBytes
    //   ... then dataBytes of tightly packed pixels, row-major, top-left origin
    constexpr u32 kTextureMagic = 0x544C4F53u;   // bytes 'S','O','L','T'
    constexpr u32 kTextureVersion = 1u;

    struct TextureBinHeader {
        u32     magic;
        u32     version;
        i32     width;
        i32     height;
        i32     channels;       // always 4; the importer expands to RGBA
        u32     format;         // TextureFormat
        u32     mipCount;       // 1 until mip generation lands
        u32     dataBytes;      // payload bytes following this header
    };

    static_assert( sizeof( TextureBinHeader ) == 32 );

    // A decoded texture in main memory, ready to hand to the renderer. Owns its
    // pixels; release with TextureAssetFree.
    struct TextureAsset {
        TextureMeta     meta;
        i32             width;
        i32             height;
        List<u8>        pixels;         // width * height * 4 bytes, RGBA8
    };

    // ---- sidecar text ------------------------------------------------------

    // Parses "key = value" lines. Blank lines and '#' comments are skipped.
    // Missing fields keep the defaults from TextureMetaDefault.
    TextureMeta TextureMetaDefault();
    bool TextureMetaParse( StringView text, TextureMeta * outMeta );
    // Serialises to the same format TextureMetaParse accepts. Round-trips.
    bool TextureMetaWrite( const TextureMeta & meta, HeapString & outText );

    // ---- whole assets ------------------------------------------------------

    // Reads the sidecar, then the payload it names from the same directory.
    bool TextureAssetLoad( StringView metaPath, TextureAsset * outAsset );
    void TextureAssetFree( TextureAsset * asset );

    // Writes both halves. The editor's importer calls this; the engine never does.
    bool TextureAssetWrite( StringView outputDirectory, StringView assetName,
                            const TextureMeta & meta,
                            const void * pixels, i32 width, i32 height );

} // namespace sol
