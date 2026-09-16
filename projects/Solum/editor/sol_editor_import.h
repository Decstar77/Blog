#pragma once

#include "sol_asset.h"
#include "sol_defines.h"
#include "sol_string.h"

namespace sol {

    // Decodes sourcePath (any format stb_image supports) and writes
    // <outputDirectory>/<assetName>.stex + .meta. Always expands to RGBA8,
    // even if the source has no alpha channel, because the engine contract
    // requires channels == 4.
    bool ImportTexture( StringView sourcePath, StringView outputDirectory, StringView assetName, TextureFormat format, TextureFilter filter, TextureWrap wrap );

} // namespace sol
