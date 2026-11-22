#include "Tensor.h"

struct Layer {
};

struct Network {
};

void nn_zero_grad( Network & nn );
Tensor nn_forward( Network & nn, const Tensor & inputs );
void nn_backward( Network & nn, const Tensor & fGrads );
void nn_update( Network & nn, float lr );