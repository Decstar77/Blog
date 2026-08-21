#include "nerf_data.h"

#include <cstdio>

#include "json.hpp"

using json = nlohmann::json;

int main() {
    // Smoke test that both vendored libraries are wired up.
    const json parsed = json::parse( R"({ "camera_angle_x": 0.69, "frames": [ { "file_path": "./train/r_0" } ] })" );
    printf( "json: camera_angle_x = %f, frames = %zu\n",
            parsed[ "camera_angle_x" ].get<nerf::f64>(),
            parsed[ "frames" ].size() );

    nerf::Image image = nerf::ReadEntireImage( "test.png" );
    if( image.pixels != nullptr ) {
        printf( "stb_image: loaded %dx%d, %d channels\n", image.width, image.height, image.channels );
        nerf::FreeImage( &image );
    }
    else {
        printf( "stb_image: no test.png to load (loader is linked)\n" );
    }

    return 0;
}
