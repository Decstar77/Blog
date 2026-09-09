#include "nerf_defines.h"

namespace nerf {
    enum ActivationFunction {
        ACTIVATION_FUNCTION_IDENTITY,
        ACTIVATION_FUNCTION_RELU,
        ACTIVATION_FUNCTION_TANH,
        ACTIVATION_FUNCTION_SIGMOID
    };

    struct NetworkMlp {
        f32                 layers[10][256];
        i32                 sizes[10];
        i32                 layerCount;

        f32                 forward[10][256];
        f32                 grads[10][256];

        ActivationFunction  actHidden;
        ActivationFunction  actOutput;
    };

    NetworkMlp *    MlpCreate( const i32 * sizes, i32 layers, ActivationFunction actHidden, ActivationFunction actOut, u32 seed = 0 );
    void            MlpDestroy( NetworkMlp * mlp );
    void            MlpForward( NetworkMlp * m, const f32 * in, f32 * out );
    void            MlpBackward( NetworkMlp * m, const f32 * dLdOut );
    void            MlpApplyGrads( NetworkMlp * m, f32 lr );
}