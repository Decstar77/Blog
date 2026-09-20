#include "cuda_render.h"

// cuda_gl_interop.h pulls in <GL/gl.h>, which on MSVC needs windows.h first.
#ifdef _WIN32
#include <windows.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cub/cub.cuh>

#include <cstdio>
#include <cmath>

constexpr float kFovY = 60.0f * kDeg2Rad;
constexpr float kNearPlane = 0.01f;
constexpr float kScreenBlur = 0.3f;
constexpr float kPowerCutoff = 10.0f;
constexpr float kMinAlpha = 1.0f / 255.0f;
constexpr float kMinTransmittance = 1.0f / 255.0f;

// One block per tile, one thread per pixel, so the tile is also the shared-memory batch size.
constexpr int kTileWidth = 16;
constexpr int kTileHeight = 16;
constexpr int kTileSize = kTileWidth * kTileHeight;

bool check( cudaError_t err, const char * what ) {
    if ( err != cudaSuccess ) {
        printf( "cuda: %s failed: %s\n", what, cudaGetErrorString( err ) );
        return false;
    }
    return true;
}

// Everything the compositing loop needs, and nothing else. Gaussian is 76 bytes (the mat3 alone is
// 36); this is 36, and it is what a shared-memory batch and the sorted-order gather actually cost.
struct SplatView {
    glm::vec2 mean;   // projected centre, in pixels
    glm::vec3 conic;  // inverse of the 2D covariance, packed (a, b, c) of [[a,b],[b,c]]
    glm::vec4 colour;
};

struct PixelCamera {
    glm::vec3 origin;  // camera position, world space
    glm::mat3 view;    // world -> view rotation
    float     cx, cy;  // principal point
    float     fx, fy;  // focal length, in pixels
};

/*
===================
===================
*/
static PixelCamera make_pixel_camera( glm::vec3 position, glm::mat3 rotation, int width, int height ) {
    PixelCamera cam;
    cam.origin = position;
    cam.view = glm::transpose( rotation );
    cam.cx = 0.5f * float( width );
    cam.cy = 0.5f * float( height );
    cam.fy = 0.5f * float( height ) / tanf( 0.5f * kFovY ); // Pinhole focal length in pixels for a fixed vertical FOV.
    cam.fx = cam.fy;
    return cam;
}

/*
===================
===================
*/
__device__ inline glm::mat3 gaussian_covariance( const Gaussian & g ) {
    // 3D covariance Sigma = R S S^T R^T for a splat.
    glm::mat3 s( 0.0f );
    s[0][0] = g.scale.x;
    s[1][1] = g.scale.y;
    s[2][2] = g.scale.z;

    const glm::mat3 m = g.rotation * s;
    return m * glm::transpose( m );
}

// ---------------------------------------------------------------------------
// Pass 1: project every splat once, and count the tiles it lands on.
// ---------------------------------------------------------------------------

__global__ void preprocess_splats( const Gaussian * gaussians, int count, PixelCamera cam, glm::ivec2 grid,
                                   SplatView * views, float * depths, glm::uvec4 * rects, u32 * touched ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= count ) {
        return;
    }

    touched[i] = 0;

    // EWA: project the mean, then push the 3D covariance through a local affine approximation of the
    // projection to get a 2D conic the render pass can evaluate with three multiplies.
    const Gaussian g = gaussians[i];
    const glm::vec3 t = cam.view * ( g.position - cam.origin );
    const float depth = -t.z;
    if ( depth < kNearPlane ) {
        return;
    }

    const float sx = cam.fx * t.x / depth + cam.cx;
    const float sy = cam.fy * t.y / depth + cam.cy;

    const glm::mat3 cov = cam.view * gaussian_covariance( g ) * glm::transpose( cam.view );

    // Rows of the projection Jacobian at t, i.e. d(screen)/d(view).
    const glm::vec3 j0( cam.fx / depth, 0.0f, cam.fx * t.x / ( depth * depth ) );
    const glm::vec3 j1( 0.0f, cam.fy / depth, cam.fy * t.y / ( depth * depth ) );

    // 2D covariance J * cov * J^T, plus a small blur so sub-pixel splats stay wide enough to hit a sample.
    const glm::vec3 cj0 = cov * j0;
    const glm::vec3 cj1 = cov * j1;
    const float a = glm::dot( j0, cj0 ) + kScreenBlur;
    const float b = glm::dot( j0, cj1 );
    const float c = glm::dot( j1, cj1 ) + kScreenBlur;

    const float det = a * c - b * b;
    if ( det <= 0.0f ) {
        return;
    }

    // Screen-space extent: the larger eigenvalue of [[a,b],[b,c]] is the variance along the splat's long
    // axis, so 3 sigma of it bounds the visible footprint in every direction.
    const float mid = 0.5f * ( a + c );
    const float lambda = mid + sqrtf( fmaxf( 0.0f, mid * mid - det ) );
    const float radius = ceilf( 3.0f * sqrtf( lambda ) );

    const int min_x = max( 0, int( floorf( ( sx - radius ) / kTileWidth ) ) );
    const int min_y = max( 0, int( floorf( ( sy - radius ) / kTileHeight ) ) );
    const int max_x = min( grid.x, int( ceilf( ( sx + radius ) / kTileWidth ) ) );
    const int max_y = min( grid.y, int( ceilf( ( sy + radius ) / kTileHeight ) ) );
    if ( max_x <= min_x || max_y <= min_y ) {
        return;
    }

    SplatView view;
    view.mean = glm::vec2( sx, sy );
    view.conic = glm::vec3( c / det, -b / det, a / det );
    view.colour = g.colour;

    views[i] = view;
    depths[i] = depth;
    rects[i] = glm::uvec4( min_x, min_y, max_x, max_y );
    touched[i] = u32( ( max_x - min_x ) * ( max_y - min_y ) );
}

// ---------------------------------------------------------------------------
// Pass 2: one (tile, splat) entry per overlap, keyed so a single radix sort groups
// by tile and orders by depth within each tile at the same time.
// ---------------------------------------------------------------------------

__global__ void duplicate_splats( int count, glm::ivec2 grid, const float * depths, const glm::uvec4 * rects,
                                  const u32 * offsets, u64 * keys, u32 * values ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= count ) {
        return;
    }

    // offsets is an inclusive scan of the per-splat tile counts, so this splat's entries end at its own
    // element and start at the previous one.
    const u32 end = offsets[i];
    const u32 start = i > 0 ? offsets[i - 1] : 0u;
    if ( end == start ) {
        return;
    }

    // Depth is above kNearPlane here, and positive floats keep their ordering when compared as raw bits.
    const u32 depth_bits = __float_as_uint( depths[i] );

    const glm::uvec4 rect = rects[i];
    u32 slot = start;
    for ( u32 y = rect.y; y < rect.w; y++ ) {
        for ( u32 x = rect.x; x < rect.z; x++ ) {
            keys[slot] = ( u64( y * u32( grid.x ) + x ) << 32 ) | u64( depth_bits );
            values[slot] = u32( i );
            slot++;
        }
    }
}

__global__ void identify_tile_ranges( int total, const u64 * keys, glm::uvec2 * ranges ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= total ) {
        return;
    }

    const u32 tile = u32( keys[i] >> 32 );
    if ( i == 0 ) {
        ranges[tile].x = 0;
    } else {
        const u32 prev = u32( keys[i - 1] >> 32 );
        if ( prev != tile ) {
            ranges[prev].y = u32( i );
            ranges[tile].x = u32( i );
        }
    }

    if ( i == total - 1 ) {
        ranges[tile].y = u32( total );
    }
}

// ---------------------------------------------------------------------------
// Pass 3: composite. One block per tile, walking that tile's slice of the sorted
// list front to back in shared-memory batches.
// ---------------------------------------------------------------------------

__global__ void render_tiles( cudaSurfaceObject_t surface, int width, int height, const glm::uvec2 * ranges,
                              const u32 * values, const SplatView * views ) {
    __shared__ glm::vec2 s_mean[kTileSize];
    __shared__ glm::vec3 s_conic[kTileSize];
    __shared__ glm::vec4 s_colour[kTileSize];

    const int tile = blockIdx.y * gridDim.x + blockIdx.x;
    const int x = blockIdx.x * kTileWidth + threadIdx.x;
    const int y = blockIdx.y * kTileHeight + threadIdx.y;
    const int lane = threadIdx.y * kTileWidth + threadIdx.x;

    const bool inside = x < width && y < height;
    const float px = float( x ) + 0.5f;
    const float py = float( y ) + 0.5f;

    const glm::uvec2 range = ranges[tile];
    const int to_do = int( range.y ) - int( range.x );

    glm::vec3 colour( 0.0f );
    float transmittance = 1.0f;

    // A thread outside the framebuffer, or one that has saturated, stops accumulating but still has to
    // reach every barrier, so the exit test is block-wide rather than a per-thread break.
    bool done = !inside;

    for ( int i = 0; i < to_do; i += kTileSize ) {
        if ( __syncthreads_count( done ) == kTileSize ) {
            break;
        }

        const int fetch = i + lane;
        if ( fetch < to_do ) {
            const SplatView view = views[values[range.x + fetch]];
            s_mean[lane] = view.mean;
            s_conic[lane] = view.conic;
            s_colour[lane] = view.colour;
        }
        __syncthreads();

        if ( !done ) {
            const int batch = min( kTileSize, to_do - i );
            for ( int j = 0; j < batch; j++ ) {
                const glm::vec2 d = s_mean[j] - glm::vec2( px, py );
                const glm::vec3 conic = s_conic[j];

                // Mahalanobis distance against the conic: -0.5 * d^T Sigma^-1 d.
                const float power = -0.5f * ( conic.x * d.x * d.x + conic.z * d.y * d.y ) - conic.y * d.x * d.y;
                if ( power < -kPowerCutoff ) {
                    continue;
                }

                const glm::vec4 c = s_colour[j];
                const float alpha = fminf( 0.99f, c.a * __expf( power ) );
                if ( alpha < kMinAlpha ) {
                    continue;
                }

                colour += transmittance * alpha * glm::vec3( c );
                transmittance *= 1.0f - alpha;
                if ( transmittance < kMinTransmittance ) {
                    done = true;
                    break;
                }
            }
        }

        // The next iteration overwrites the batch, so no thread may run ahead into it.
        __syncthreads();
    }

    if ( !inside ) {
        return;
    }

    uchar4 pixel;
    pixel.x = (unsigned char) ( 255.0f * fminf( colour.x, 1.0f ) );
    pixel.y = (unsigned char) ( 255.0f * fminf( colour.y, 1.0f ) );
    pixel.z = (unsigned char) ( 255.0f * fminf( colour.z, 1.0f ) );
    pixel.w = (unsigned char) ( 255.0f * fminf( 1.0f - transmittance, 1.0f ) );

    surf2Dwrite( pixel, surface, x * int( sizeof( uchar4 ) ), y );
}

// ---------------------------------------------------------------------------
// Device-side state. Every buffer grows by capacity and is never shrunk, so a
// steady-state frame does no allocation.
// ---------------------------------------------------------------------------

struct DeviceBuffer {
    void * ptr;
    u64    cap;
};

static bool buffer_reserve( DeviceBuffer * buffer, u64 bytes ) {
    if ( bytes <= buffer->cap ) {
        return true;
    }

    // Exact on the first allocation, then 1.5x, so a scene that grows a splat at a time does not
    // reallocate every frame without rounding a 300 MB buffer up to 512.
    u64 want = buffer->cap + buffer->cap / 2;
    if ( want < bytes ) {
        want = bytes;
    }

    void * ptr = nullptr;
    if ( !check( cudaMalloc( &ptr, want ), "cudaMalloc" ) ) {
        return false;
    }

    cudaFree( buffer->ptr );
    buffer->ptr = ptr;
    buffer->cap = want;
    return true;
}

static void buffer_free( DeviceBuffer * buffer ) {
    cudaFree( buffer->ptr );
    buffer->ptr = nullptr;
    buffer->cap = 0;
}

template <typename _type_>
static _type_ * buffer_as( DeviceBuffer * buffer ) {
    return (_type_ *) buffer->ptr;
}

static cudaGraphicsResource * g_resource = nullptr;

static DeviceBuffer g_gaussians = {};  // Gaussian[count], the scene as uploaded
static DeviceBuffer g_views = {};      // SplatView[count]
static DeviceBuffer g_depths = {};     // float[count]
static DeviceBuffer g_rects = {};      // uvec4[count], tile-space bounds
static DeviceBuffer g_touched = {};    // u32[count], tiles each splat covers
static DeviceBuffer g_offsets = {};    // u32[count], inclusive scan of g_touched
static DeviceBuffer g_keys_in = {};    // u64[total]
static DeviceBuffer g_keys_out = {};
static DeviceBuffer g_values_in = {};  // u32[total]
static DeviceBuffer g_values_out = {};
static DeviceBuffer g_ranges = {};     // uvec2[tiles]
static DeviceBuffer g_scratch = {};    // cub temp storage

static int g_uploaded_count = 0;       // splats currently resident in g_gaussians

/*
===================
===================
*/
static int tile_id_bits( int tiles ) {
    int bits = 1;
    while ( ( 1 << bits ) < tiles ) {
        bits++;
    }
    return bits;
}

/*
===================
===================
*/
bool cuda_render_init( uint32_t gl_texture ) {
    int device_count = 0;
    if ( !check( cudaGetDeviceCount( &device_count ), "cudaGetDeviceCount" ) || device_count == 0 ) {
        printf( "cuda: no capable device found\n" );
        return false;
    }

    cudaDeviceProp props = {};
    if ( check( cudaGetDeviceProperties( &props, 0 ), "cudaGetDeviceProperties" ) ) {
        printf( "cuda: %s (sm_%d%d)\n", props.name, props.major, props.minor );
    }

    cudaGraphicsResource * resource = nullptr;
    if ( !check( cudaGraphicsGLRegisterImage( &resource, gl_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore ), "cudaGraphicsGLRegisterImage" ) ) {
        return false;
    }

    if ( g_resource ) {
        cudaGraphicsUnregisterResource( g_resource );
    }
    g_resource = resource;
    return true;
}

/*
===================
===================
*/
void cuda_render_shutdown() {
    if ( g_resource ) {
        cudaGraphicsUnregisterResource( g_resource );
        g_resource = nullptr;
    }

    buffer_free( &g_gaussians );
    buffer_free( &g_views );
    buffer_free( &g_depths );
    buffer_free( &g_rects );
    buffer_free( &g_touched );
    buffer_free( &g_offsets );
    buffer_free( &g_keys_in );
    buffer_free( &g_keys_out );
    buffer_free( &g_values_in );
    buffer_free( &g_values_out );
    buffer_free( &g_ranges );
    buffer_free( &g_scratch );

    g_uploaded_count = 0;
}

/*
===================
===================
*/
static bool reserve_scratch( int count, int total, int end_bit ) {
    size_t scan_bytes = 0;
    cub::DeviceScan::InclusiveSum( nullptr, scan_bytes, buffer_as<u32>( &g_touched ), buffer_as<u32>( &g_offsets ), count );

    size_t sort_bytes = 0;
    if ( total > 0 ) {
        cub::DeviceRadixSort::SortPairs( nullptr, sort_bytes, buffer_as<u64>( &g_keys_in ), buffer_as<u64>( &g_keys_out ),
                                         buffer_as<u32>( &g_values_in ), buffer_as<u32>( &g_values_out ), total, 0, end_bit );
    }

    return buffer_reserve( &g_scratch, scan_bytes > sort_bytes ? scan_bytes : sort_bytes );
}

/*
===================
===================
*/
void cuda_render_frame( Scene * scene, int width, int height, float time ) {
    (void) time;

    if ( !g_resource || width <= 0 || height <= 0 ) {
        return;
    }

    const int count = scene->gaussians.count;
    const glm::ivec2 grid( ( width + kTileWidth - 1 ) / kTileWidth, ( height + kTileHeight - 1 ) / kTileHeight );
    const int tiles = grid.x * grid.y;

    // A loaded capture is hundreds of megabytes, so the upload is driven by the scene's dirty flag
    // rather than done every frame. The count check is a backstop for a caller that edits the list and
    // forgets to set it; it cannot catch an in-place edit that leaves the count alone.
    if ( count > 0 && ( scene->gaussians_dirty || g_uploaded_count != count ) ) {
        if ( !buffer_reserve( &g_gaussians, u64( count ) * sizeof( Gaussian ) ) ) {
            return;
        }
        if ( !check( cudaMemcpy( g_gaussians.ptr, scene->gaussians.data, u64( count ) * sizeof( Gaussian ), cudaMemcpyHostToDevice ), "cudaMemcpy scene" ) ) {
            return;
        }

        g_uploaded_count = count;
        scene->gaussians_dirty = false;
    }

    const bool sized = buffer_reserve( &g_views, u64( count ) * sizeof( SplatView ) ) &&
                       buffer_reserve( &g_depths, u64( count ) * sizeof( float ) ) &&
                       buffer_reserve( &g_rects, u64( count ) * sizeof( glm::uvec4 ) ) &&
                       buffer_reserve( &g_touched, u64( count ) * sizeof( u32 ) ) &&
                       buffer_reserve( &g_offsets, u64( count ) * sizeof( u32 ) ) &&
                       buffer_reserve( &g_ranges, u64( tiles ) * sizeof( glm::uvec2 ) );
    if ( !sized ) {
        return;
    }

    // Tiles no splat lands on are never written by identify_tile_ranges, so they have to start empty.
    if ( !check( cudaMemset( g_ranges.ptr, 0, u64( tiles ) * sizeof( glm::uvec2 ) ), "cudaMemset ranges" ) ) {
        return;
    }

    const PixelCamera cam = make_pixel_camera( scene->camera.position, scene->camera.rotation, width, height );

    const int threads = 256;
    const int splat_blocks = ( count + threads - 1 ) / threads;

    int total = 0;
    if ( count > 0 ) {
        preprocess_splats<<<splat_blocks, threads>>>( buffer_as<Gaussian>( &g_gaussians ), count, cam, grid,
                                                      buffer_as<SplatView>( &g_views ), buffer_as<float>( &g_depths ),
                                                      buffer_as<glm::uvec4>( &g_rects ), buffer_as<u32>( &g_touched ) );
        if ( !check( cudaGetLastError(), "preprocess_splats launch" ) ) {
            return;
        }

        if ( !reserve_scratch( count, 0, 64 ) ) {
            return;
        }

        size_t scratch_bytes = g_scratch.cap;
        if ( !check( cub::DeviceScan::InclusiveSum( g_scratch.ptr, scratch_bytes, buffer_as<u32>( &g_touched ),
                                                    buffer_as<u32>( &g_offsets ), count ), "cub InclusiveSum" ) ) {
            return;
        }

        // The last element of the scan is the total number of (tile, splat) entries to sort.
        u32 host_total = 0;
        if ( !check( cudaMemcpy( &host_total, buffer_as<u32>( &g_offsets ) + ( count - 1 ), sizeof( u32 ), cudaMemcpyDeviceToHost ), "cudaMemcpy total" ) ) {
            return;
        }
        total = int( host_total );
    }

    if ( total > 0 ) {
        const bool keyed = buffer_reserve( &g_keys_in, u64( total ) * sizeof( u64 ) ) &&
                           buffer_reserve( &g_keys_out, u64( total ) * sizeof( u64 ) ) &&
                           buffer_reserve( &g_values_in, u64( total ) * sizeof( u32 ) ) &&
                           buffer_reserve( &g_values_out, u64( total ) * sizeof( u32 ) );
        if ( !keyed ) {
            return;
        }

        duplicate_splats<<<splat_blocks, threads>>>( count, grid, buffer_as<float>( &g_depths ), buffer_as<glm::uvec4>( &g_rects ),
                                                     buffer_as<u32>( &g_offsets ), buffer_as<u64>( &g_keys_in ), buffer_as<u32>( &g_values_in ) );
        if ( !check( cudaGetLastError(), "duplicate_splats launch" ) ) {
            return;
        }

        // Only the bits the tile id actually occupies need sorting above the 32 depth bits.
        const int end_bit = 32 + tile_id_bits( tiles );
        if ( !reserve_scratch( count, total, end_bit ) ) {
            return;
        }

        size_t scratch_bytes = g_scratch.cap;
        if ( !check( cub::DeviceRadixSort::SortPairs( g_scratch.ptr, scratch_bytes, buffer_as<u64>( &g_keys_in ), buffer_as<u64>( &g_keys_out ),
                                                      buffer_as<u32>( &g_values_in ), buffer_as<u32>( &g_values_out ), total, 0, end_bit ), "cub SortPairs" ) ) {
            return;
        }

        const int range_blocks = ( total + threads - 1 ) / threads;
        identify_tile_ranges<<<range_blocks, threads>>>( total, buffer_as<u64>( &g_keys_out ), buffer_as<glm::uvec2>( &g_ranges ) );
        if ( !check( cudaGetLastError(), "identify_tile_ranges launch" ) ) {
            return;
        }
    }

    if ( !check( cudaGraphicsMapResources( 1, &g_resource ), "cudaGraphicsMapResources" ) ) {
        return;
    }

    cudaArray_t array = nullptr;
    if ( check( cudaGraphicsSubResourceGetMappedArray( &array, g_resource, 0, 0 ), "cudaGraphicsSubResourceGetMappedArray" ) ) {
        cudaResourceDesc desc = {};
        desc.resType = cudaResourceTypeArray;
        desc.res.array.array = array;

        cudaSurfaceObject_t surface = 0;
        if ( check( cudaCreateSurfaceObject( &surface, &desc ), "cudaCreateSurfaceObject" ) ) {
            const dim3 block( kTileWidth, kTileHeight );
            const dim3 blocks( grid.x, grid.y );
            render_tiles<<<blocks, block>>>( surface, width, height, buffer_as<glm::uvec2>( &g_ranges ),
                                             buffer_as<u32>( &g_values_out ), buffer_as<SplatView>( &g_views ) );
            check( cudaGetLastError(), "render_tiles launch" );

            cudaDestroySurfaceObject( surface );
        }
    }

    check( cudaGraphicsUnmapResources( 1, &g_resource ), "cudaGraphicsUnmapResources" );
}
