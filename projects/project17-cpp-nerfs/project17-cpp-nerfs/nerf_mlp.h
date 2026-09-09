#pragma once
#include "nerf_defines.h"

namespace nerf {
    constexpr i32 kMlpMaxLayers = 10;
    constexpr i32 kMlpMaxWidth = 256;

    enum ActivationFunction {
        ACTIVATION_FUNCTION_IDENTITY,
        ACTIVATION_FUNCTION_RELU,
        ACTIVATION_FUNCTION_TANH,
        ACTIVATION_FUNCTION_SIGMOID
    };

    enum OptimizerType {
        OPTIMIZER_SGD,
        OPTIMIZER_ADAM
    };

    // A plain fully connected net. sizes[0] is the input width, sizes[layerCount - 1] the output
    // width, so there are layerCount - 1 weight matrices. Everything is single sample, no batching:
    // MlpForward stashes the activations it needs, MlpBackward accumulates into the gradient
    // buffers, and MlpApplyGrads takes the step and clears them. That means you run one
    // forward/backward pair per sample and step once the mini batch is done; the accumulated
    // gradients are averaged over the batch, so the learning rate does not shift with batch size.
    struct NetworkMlp {
        i32                 sizes[kMlpMaxLayers];
        i32                 layerCount;

        // weights[l] is sizes[l + 1] rows of sizes[l] columns, row major, so weights[l][o * in + i]
        // is the weight from input i to output o. Valid for l in [0, layerCount - 1).
        f32 *               weights[kMlpMaxLayers];
        f32 *               biases[kMlpMaxLayers];
        f32 *               weightGrads[kMlpMaxLayers];
        f32 *               biasGrads[kMlpMaxLayers];

        f32                 activations[kMlpMaxLayers][kMlpMaxWidth];    // post activation, [0] is the input
        f32                 preActivations[kMlpMaxLayers][kMlpMaxWidth]; // pre activation, valid for l >= 1
        f32                 deltas[kMlpMaxLayers][kMlpMaxWidth];         // dL/dPreActivation

        ActivationFunction  actHidden;
        ActivationFunction  actOutput;

        // Adam state, allocated only by MlpSetOptimizerAdam. First and second moment per parameter,
        // plus the running beta^t used for bias correction.
        OptimizerType       optimizer;
        f32 *               weightM[kMlpMaxLayers];
        f32 *               weightV[kMlpMaxLayers];
        f32 *               biasM[kMlpMaxLayers];
        f32 *               biasV[kMlpMaxLayers];
        f32                 beta1;
        f32                 beta2;
        f32                 adamEps;
        f32                 beta1Pow;
        f32                 beta2Pow;

        i32                 gradAccumCount; // MlpBackward calls since the last step, the batch size
    };

    NetworkMlp *    MlpCreate( const i32 * sizes, i32 layers, ActivationFunction actHidden, ActivationFunction actOut, u32 seed = 0 );
    void            MlpDestroy( NetworkMlp * mlp );

    // Switches the net from plain SGD to Adam and allocates the moment buffers. Call once after
    // MlpCreate. The defaults are the usual NeRF settings.
    void            MlpSetOptimizerAdam( NetworkMlp * m, f32 beta1 = 0.9f, f32 beta2 = 0.999f, f32 eps = 1e-8f );

    void            MlpForward( NetworkMlp * m, const f32 * in, f32 * out );

    // Mean squared error between the outputs the last MlpForward left in the net and target,
    // averaged over the output width. Returns the loss and writes dL/dOut, which is exactly what
    // MlpBackward wants, so the usual sequence is forward -> loss -> backward. target and dLdOut
    // are both sizes[layerCount - 1] wide; dLdOut may be null if you only want the number.
    f32             MlpLossMSE( NetworkMlp * m, const f32 * target, f32 * dLdOut );

    void            MlpBackward( NetworkMlp * m, const f32 * dLdOut );
    void            MlpApplyGrads( NetworkMlp * m, f32 lr );
    void            MlpZeroGrads( NetworkMlp * m );
}
