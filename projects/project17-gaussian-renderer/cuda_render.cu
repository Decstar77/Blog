#include "cuda_render.h"

// cuda_gl_interop.h pulls in <GL/gl.h>, which on MSVC needs windows.h first.
#ifdef _WIN32
#include <windows.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdio>

cudaGraphicsResource * g_resource = nullptr;

bool check( cudaError_t err, const char * what ) {
    if ( err != cudaSuccess ) {
        printf( "cuda: %s failed: %s\n", what, cudaGetErrorString( err ) );
        return false;
    }
    return true;
}

__global__ void fill_kernel( cudaSurfaceObject_t surface, int width, int height, float time ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( x >= width || y >= height ) {
        return;
    }

    const float u = float( x ) / float( width );
    const float v = float( y ) / float( height );

    uchar4 pixel;
    pixel.x = (unsigned char) ( 255.0f * u );
    pixel.y = (unsigned char) ( 255.0f * v );
    pixel.z = (unsigned char) ( 255.0f * ( 0.5f + 0.5f * __sinf( time ) ) );
    pixel.w = 255;

    surf2Dwrite( pixel, surface, x * int( sizeof( uchar4 ) ), y );
}

__global__ void render_gaussian( cudaSurfaceObject_t surface, Gaussian * gaussians, int count, glm::vec3 cp, glm::mat3 cv,int width, int height ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = 0; i < count; i++) {
        
    }
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

static Gaussian * d_gaussians = nullptr;
static i32 d_count = 0;

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
            render_gaussian<<<grid, threads>>>( surface, d_gaussians, scene->camera.position, scene->camera.rotation, width, height );
            check( cudaGetLastError(), "fill_kernel launch" );

            cudaDestroySurfaceObject( surface );
        }
    }

    check( cudaGraphicsUnmapResources( 1, &g_resource ), "cudaGraphicsUnmapResources" );
}
