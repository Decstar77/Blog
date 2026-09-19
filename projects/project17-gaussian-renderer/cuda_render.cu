#include "cuda_render.h"

// cuda_gl_interop.h pulls in <GL/gl.h>, which on MSVC needs windows.h first.
#ifdef _WIN32
#include <windows.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdio>

constexpr float kFovY = 60.0f * kDeg2Rad;
constexpr float kNearPlane = 0.01f;
constexpr float kScreenBlur = 0.3f;
constexpr float kPowerCutoff = 10.0f;

static cudaGraphicsResource *   g_resource = nullptr;
static Gaussian *               d_gaussians = nullptr;
static i32                      d_count = 0;

bool check( cudaError_t err, const char * what ) {
    if ( err != cudaSuccess ) {
        printf( "cuda: %s failed: %s\n", what, cudaGetErrorString( err ) );
        return false;
    }
    return true;
}

__device__ inline glm::mat3 gaussian_covariance( const Gaussian & g ) {
    // 3D covariance Sigma = R S S^T R^T for a splat.
    glm::mat3 s( 0.0f );
    s[0][0] = g.scale.x;
    s[1][1] = g.scale.y;
    s[2][2] = g.scale.z;

    const glm::mat3 m = g.rotation * s;
    return m * glm::transpose( m );
}

__device__ inline glm::mat3 gaussian_inverse_transform( const Gaussian & g ) {
    // Inverse of the R*S that shapes a splat, so a world offset from the mean becomes a local one whose
    // squared length is the Gaussian's exponent: q(x) = |N (x - mu)|^2, N = diag(1/scale) * R^T.
    // Assumes g.rotation is orthonormal (R^-1 == R^T). glm is column-major, so dividing each column by
    // the scale vector scales row r by 1/scale[r], which is the left-multiply we want.
    glm::mat3 n = glm::transpose( g.rotation );
    n[0] /= g.scale;
    n[1] /= g.scale;
    n[2] /= g.scale;
    return n;
}

struct PixelCamera {
    glm::vec3 origin;  // camera position, world space
    glm::mat3 view;    // world -> view rotation
    glm::vec3 ray;     // world-space direction through this pixel centre
    float     x, y;    // pixel centre, in pixels
    float     cx, cy;  // principal point
    float     fx, fy;  // focal length, in pixels
};

__device__ inline float splat_alpha_ema( const Gaussian & g, const PixelCamera & cam ) {
    // EWA: project the mean, then push the 3D covariance through a local affine approximation of the projection to get a 2D conic to evaluate at the pixel centre.
    const glm::vec3 t = cam.view * ( g.position - cam.origin );
    const float depth = -t.z;
    if ( depth < kNearPlane ) {
        return 0.0f;
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
        return 0.0f;
    }

    // Mahalanobis distance with the inverse of [[a,b],[b,c]].
    const float dx = cam.x - sx;
    const float dy = cam.y - sy;
    const float power = -0.5f * ( c * dx * dx - 2.0f * b * dx * dy + a * dy * dy ) / det;
    if ( power < -kPowerCutoff ) {
        return 0.0f;
    }

    return fminf( 0.99f, g.colour.a * __expf( power ) );
}

__device__ inline float splat_alpha_ray( const Gaussian & g, const PixelCamera & cam ) {
    // Ray: intersect the real camera ray with the 3D Gaussian, no projection and no affine approximation.
    // In the splat's local frame the exponent along the ray is a plain quadratic q(t) = a t^2 + 2 b t + c,
    // minimised at t = -b/a. Taking the density at that closest approach is the max particle response;
    // the alternative is the closed-form line integral, the same peak scaled by sqrt( 2 pi / a ).
    const glm::mat3 n = gaussian_inverse_transform( g );
    const glm::vec3 o = n * ( cam.origin - g.position );
    const glm::vec3 d = n * cam.ray;

    const float a = glm::dot( d, d );
    if ( a <= 0.0f ) {
        return 0.0f;
    }

    const float b = glm::dot( d, o );
    const float c = glm::dot( o, o );

    // -b/a is the closest-approach parameter and the view-space depth there, so this culls what is behind us.
    if ( -b / a < kNearPlane ) {
        return 0.0f;
    }

    const float power = -0.5f * ( c - b * b / a );
    if ( power < -kPowerCutoff ) {
        return 0.0f;
    }

    return fminf( 0.99f, g.colour.a * __expf( power ) );
}

__global__ void render_gaussian( cudaSurfaceObject_t surface, Gaussian * gaussians, int count, glm::vec3 cp, glm::mat3 cv, int width, int height ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( x >= width || y >= height ) {
        return;
    }

    PixelCamera cam = {};
    cam.origin = cp;
    cam.view = glm::transpose( cv );
    cam.x = float( x ) + 0.5f;
    cam.y = float( y ) + 0.5f;
    cam.cx = 0.5f * float( width );
    cam.cy = 0.5f * float( height );
    cam.fy = 0.5f * float( height ) / tanf( 0.5f * kFovY ); // Pinhole focal length in pixels for a fixed vertical FOV.
    cam.fx = cam.fy;
    cam.ray = cv * glm::vec3( ( cam.x - cam.cx ) / cam.fx, ( cam.y - cam.cy ) / cam.fy, -1.0f );

    glm::vec3 colour( 0.0f );
    float transmittance = 1.0f;
    for ( int i = 0; i < count; i++ ) {
        const Gaussian g = gaussians[i];
        const float alpha = splat_alpha_ray( g, cam );
        if ( alpha < 1.0f / 255.0f ) {
            continue;
        }

        colour += transmittance * alpha * glm::vec3( g.colour );
        transmittance *= 1.0f - alpha;
        if ( transmittance < 1.0f / 255.0f ) {
            break;
        }
    }

    uchar4 pixel;
    pixel.x = (unsigned char) ( 255.0f * fminf( colour.x, 1.0f ) );
    pixel.y = (unsigned char) ( 255.0f * fminf( colour.y, 1.0f ) );
    pixel.z = (unsigned char) ( 255.0f * fminf( colour.z, 1.0f ) );
    pixel.w = (unsigned char) ( 255.0f * fminf( 1.0f - transmittance, 1.0f ) );

    surf2Dwrite( pixel, surface, x * int( sizeof( uchar4 ) ), y );
}

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

    cuda_render_shutdown();
    g_resource = resource;
    return true;
}

void cuda_render_shutdown() {
    if ( g_resource ) {
        cudaGraphicsUnregisterResource( g_resource );
        g_resource = nullptr;
    }
}

void cuda_render_frame( Scene * scene, int width, int height, float time ) {
    if ( !g_resource || width <= 0 || height <= 0 ) {
        return;
    }

    if ( d_count != scene->gaussians.count ) {
        if ( d_gaussians != nullptr ) {
            check( cudaFree( d_gaussians ), "cudaFree" );
            d_count = 0;
        }

        d_count = scene->gaussians.count;
        if ( check( cudaMalloc( &d_gaussians, d_count * sizeof( Gaussian ) ), "cudaMalloc" ) == false ) {
            return;
        }

        if ( check( cudaMemcpy( d_gaussians, scene->gaussians.data, d_count * sizeof( Gaussian ), cudaMemcpyHostToDevice ), "cudaMemcpy" ) == false ) {
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
            const dim3 threads( 16, 16 );
            const dim3 grid( ( width + threads.x - 1 ) / threads.x, ( height + threads.y - 1 ) / threads.y );
            render_gaussian<<<grid, threads>>>( surface, d_gaussians, d_count, scene->camera.position, scene->camera.rotation, width, height );
            check( cudaGetLastError(), "render_gaussian launch" );
            check( cudaDestroySurfaceObject( surface ), "cudaDestroySurfaceObject" );
        }
    }

    check( cudaGraphicsUnmapResources( 1, &g_resource ), "cudaGraphicsUnmapResources" );
}
