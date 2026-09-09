#include "nerf_mlp.h"


namespace nerf {
    NetworkMlp * MlpCreate( const i32 * sizes, const i32 layers, ActivationFunction actHidden, ActivationFunction actOut, u32 seed ) {
        NetworkMlp * mlp = new NetworkMlp();
        mlp->layerCount = layers;
        mlp->actHidden = actHidden;
        mlp->actOutput = actOut;
        for ( i32 i = 0; i < mlp->layerCount; i++ ) {
            mlp->sizes[i] = sizes[i];
        }
        return mlp;
    }
    
    void MlpDestroy( NetworkMlp * mlp ) {
        delete mlp;
    }

    static void ComputeActivationFunction(ActivationFunction func, f32 * in, i32 size ) {
        switch (func) {
            case ACTIVATION_FUNCTION_IDENTITY:
                return;
            case ACTIVATION_FUNCTION_RELU: {
                for (i32 i = 0; i < size; i++) {
                    in[i] = Max( 0, in[i] );
                }
            } break;
        }
    }

    void MlpForward( NetworkMlp * m, const f32 * in, f32 * out ) {
        for ( i32 l = 0; l < m->layerCount; l++ ) {
            for ( i32 i = 0; i < m->sizes[l]; i++ ) {

            }
        }
    }

    void MlpBackward( NetworkMlp * m, const f32 * dLdOut ) {
        
    }

    void MlpApplyGrads( NetworkMlp * m, f32 lr ) {

    }
}
