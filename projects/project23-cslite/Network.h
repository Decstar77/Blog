#pragma once

#include "Tensor.h"
#include "CSGame.h"

#include <vector>

/*
===============================================================================
===============================================================================
*/
enum class Activation {
    Tanh,
    Linear,
};

/*
===============================================================================
===============================================================================
*/
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

/*
===============================================================================
===============================================================================
*/
struct Network {
    std::vector<Layer> layers;
};

Network         nn_create( const std::vector<int> & sizes );
void            nn_free( Network & nn );
void            nn_zero_grad( Network & nn );
const Tensor &  nn_forward( Network & nn, const Tensor & inputs );    // returned tensor is owned by nn, overwritten on next forward
void            nn_backward( Network & nn, const Tensor & fGrads );   // fGrads = dLoss/d(output)
void            nn_update( Network & nn, float lr );                  // W -= lr * dW
