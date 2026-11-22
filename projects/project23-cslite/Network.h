#pragma once

#include "Tensor.h"

#include <vector>

// Minimal fully connected network (MLP) for single samples, no batching.
//
// Usage per sample:
//   const Tensor & y = nn_forward( nn, x );   // caches activations inside the layers
//   nn_backward( nn, dLoss_dy );              // ADDS into dW/db, uses the cached activations
// Gradients accumulate until nn_zero_grad, so a mini-batch is:
//   nn_zero_grad -> (forward, backward) for each sample -> nn_update( lr / batchSize )
//
// nn_backward must follow the nn_forward of the same sample, because it reads the
// activations that forward cached.

enum class Activation {
    Tanh,
    Linear,
};

struct Layer {
    Activation act;

    Tensor W;           // [out, in]
    Tensor b;           // [out]
    Tensor dW;          // accumulated dLoss/dW
    Tensor db;          // accumulated dLoss/db

    // Cached by nn_forward, read by nn_backward
    Tensor input;       // x
    Tensor output;      // y = act( W x + b )

    // Written by nn_backward: dLoss/dx, handed to the previous layer
    Tensor gradInput;
};

struct Network {
    std::vector<Layer> layers;
};

// sizes = { inputs, hidden..., outputs }. Hidden layers use tanh, the last layer is linear
// (actor: raw logits, critic: raw value).
Network nn_create( const std::vector<int> & sizes );
void    nn_free( Network & nn );

void            nn_zero_grad( Network & nn );
const Tensor &  nn_forward( Network & nn, const Tensor & inputs );   // returned tensor is owned by nn, overwritten on next forward
void            nn_backward( Network & nn, const Tensor & fGrads );   // fGrads = dLoss/d(output)
void            nn_update( Network & nn, float lr );                 // W -= lr * dW
