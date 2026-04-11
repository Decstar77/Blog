#include <random>
#include <math.h>

float random_float( float min = -1.0f, float max = 1.0f ) {
    // 'static' ensures the engine is initialized only once
    static std::random_device rd;
    static std::mt19937 gen( rd() );

    std::uniform_real_distribution<float> dis( min, max );
    return dis( gen );
}

float loss_mse( float * preds, float * targets, int size ) {
    float sum = 0.0f;
    for ( int i = 0; i < size; i++ ) {
        sum += (preds[i] - targets[i]) * (preds[i] - targets[i]);
    }
    sum /= (float)size;
    return sum;
}

struct LinearLayer {
    int     input_count;
    int     neuron_count;
    int     size;
    float * weights;
    float * biases;
    float * weight_grads;
    float * bias_grads;
    float * pre_activation;
};

LinearLayer make_layer( int in_features, int out_features ) {
    LinearLayer layer = {};
    layer.input_count = in_features;
    layer.neuron_count = out_features;
    layer.size = layer.input_count * layer.neuron_count;

    layer.weights           = new float[layer.size];
    layer.weight_grads      = new float[layer.size];
    layer.pre_activation    = new float[layer.neuron_count];
    layer.biases            = new float[layer.neuron_count];
    layer.bias_grads        = new float[layer.neuron_count];

    for ( int i = 0; i < layer.size; i++ ) { layer.weights[i] = random_float(); }
    for ( int i = 0; i < layer.size; i++ ) { layer.weight_grads[i] = 0.0f; }
    for ( int i = 0; i < layer.neuron_count; i++ ) { layer.pre_activation[i] = 0.0f; }
    for ( int i = 0; i < layer.neuron_count; i++ ) { layer.biases[i] = random_float(); }
    for ( int i = 0; i < layer.neuron_count; i++ ) { layer.bias_grads[i] = 0.0f; }

    return layer;
}

void zero_grad( LinearLayer & layer ) {
    for ( int i = 0; i < layer.size; i++ ) { layer.weight_grads[i] = 0.0f; }
    for ( int i = 0; i < layer.neuron_count; i++ ) { layer.bias_grads[i] = 0.0f; }
    for ( int i = 0; i < layer.neuron_count; i++ ) { layer.pre_activation[i] = 0.0f; }
}

float * forward( const float * inputs, const LinearLayer & layer ) {
    float * result = new float[layer.neuron_count];
    for ( int neuron_index = 0; neuron_index < layer.neuron_count; neuron_index++ ) {
        float sum = 0;
        for ( int weight_index = 0; weight_index < layer.input_count; weight_index++ ) {
            sum += layer.weights[neuron_index * layer.input_count + weight_index] * inputs[weight_index];
        }
        sum += layer.biases[neuron_index];
        layer.pre_activation[neuron_index] = sum;
        result[neuron_index] = tanhf(sum);
    }
    return result;
}

float tahn_derivative( float x ) {
    x = tanh( x );
    return 1.0f - x * x;
}

float * backward( const float * upstream_grad, const float * inputs, LinearLayer & layer ) {
    // Chain upstream grad through tanh derivative once
    float * local_grad = new float[layer.neuron_count];
    for ( int i = 0; i < layer.neuron_count; i++ ) {
        local_grad[i] = upstream_grad[i] * tahn_derivative(layer.pre_activation[i] );
    }

    // dL/db = local_grad
    for ( int i = 0; i < layer.neuron_count; i++ ) {
        layer.bias_grads[i] = local_grad[i];
    }

    // dL/dW[neuron, inp] = local_grad[neuron] * inputs[inp]
    for ( int neuron = 0; neuron < layer.neuron_count; neuron++ ) {
        for ( int inp = 0; inp < layer.input_count; inp++ ) {
            layer.weight_grads[neuron * layer.input_count + inp] = local_grad[neuron] * inputs[inp];
        }
    }

    // dL/dinputs[inp] = sum over neurons of local_grad[neuron] * W[neuron, inp]
    float * downstream = new float[layer.input_count]();
    for ( int inp = 0; inp < layer.input_count; inp++ ) {
        for ( int neuron = 0; neuron < layer.neuron_count; neuron++ ) {
            downstream[inp] += local_grad[neuron] * layer.weights[neuron * layer.input_count + inp];
        }
    }

    delete[] local_grad;
    return downstream;
}

void sgd_step( LinearLayer & layer, float lr ) {
    for ( int i = 0; i < layer.size; i++ ) { layer.weights[i] -= lr * layer.weight_grads[i]; }
    for ( int i = 0; i < layer.neuron_count; i++ ) { layer.biases[i] -= lr * layer.bias_grads[i]; }
}

static const int data_size = 1000;
static float samples[data_size] = {};
static float targets[data_size] = {};

int main() {
    for ( int i = 0; i < data_size; i++ ) {
        samples[i] = random_float( -2.0f, 2.0f );
        targets[i] = samples[i] * samples[i];
    }

    LinearLayer l1 = make_layer( 1, 32 );
    LinearLayer l2 = make_layer( 32, 32 );
    LinearLayer l3 = make_layer( 32, 1 );

    for ( int i = 0; i < data_size; i++ ) {
        zero_grad( l1 );
        zero_grad( l2 );
        zero_grad( l3 );

        float * o1 = forward( &samples[i], l1 );
        float * o2 = forward( o1, l2 );
        float * o3 = forward( o2, l3 );

        float loss = loss_mse( o3, &targets[i], 1 );

        // dL/do3 = 2*(pred - target) / size, size=1 here
        float loss_grad[1] = { 2.0f * (o3[0] - targets[i]) };

        float * g3 = backward( loss_grad, o2, l3 );
        float * g2 = backward( g3, o1, l2 );
        float * g1 = backward( g2, &samples[i], l1 );

        delete[] g3;
        delete[] g2;
        delete[] g1;

        static const float lr = 0.001f;
        sgd_step( l1, lr );
        sgd_step( l2, lr );
        sgd_step( l3, lr );

        delete[] o1;
        delete[] o2;
        delete[] o3;
    }

    return 0;
}