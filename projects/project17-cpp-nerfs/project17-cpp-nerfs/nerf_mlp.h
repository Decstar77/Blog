#pragma once
#include "nerf_defines.h"

namespace nerf {
    constexpr i32 MlpMaxLayers = 10;
    constexpr i32 MlpMaxWidth = 256;

    enum ActivationFunction {
        ACTIVATION_FUNCTION_IDENTITY,
        ACTIVATION_FUNCTION_RELU,
        ACTIVATION_FUNCTION_TANH,
        ACTIVATION_FUNCTION_SIGMOID
    };

    struct NetworkMlp {
        i32                 sizes[MlpMaxLayers];
        i32                 layerCount;

        f32 *               weights[MlpMaxLayers];
        f32 *               biases[MlpMaxLayers];
        f32 *               weightGrads[MlpMaxLayers];
        f32 *               biasGrads[MlpMaxLayers];

        f32                 activations[MlpMaxLayers][MlpMaxWidth];    // post activation, [0] is the input
        f32                 preActivations[MlpMaxLayers][MlpMaxWidth]; // pre activation, valid for l >= 1
        f32                 deltas[MlpMaxLayers][MlpMaxWidth];         // dL/dPreActivation

        ActivationFunction  actHidden;
        ActivationFunction  actOutput;
    };

    NetworkMlp *    MlpCreate( const i32 * sizes, i32 layers, ActivationFunction actHidden, ActivationFunction actOut, u32 seed = 0 );
    void            MlpDestroy( NetworkMlp * mlp );
    void            MlpForward( NetworkMlp * m, const f32 * in, f32 * out );
    void            MlpBackward( NetworkMlp * m, const f32 * dLdOut );
    void            MlpApplyGrads( NetworkMlp * m, f32 lr );
    void            MlpZeroGrads( NetworkMlp * m );
}
