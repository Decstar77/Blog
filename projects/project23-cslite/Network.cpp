#include "Network.h"

#include <cassert>
#include <cmath>

Network nn_create( const std::vector<int> & sizes ) {
    assert( sizes.size() >= 2 );
    Network nn;
    for ( size_t i = 0; i + 1 < sizes.size(); i++ ) {
        int in = sizes[i];
        int out = sizes[i + 1];
        bool isLast = ( i + 2 == sizes.size() );

        Layer layer;
        layer.act = isLast ? Activation::Linear : Activation::Tanh;
        layer.W = tn_create( out, in );
        layer.b = tn_create( out );
        layer.dW = tn_create( out, in );
        layer.db = tn_create( out );
        layer.input = tn_create( in );
        layer.output = tn_create( out );
        layer.gradInput = tn_create( in );

        // Scale by 1/sqrt(in) so activations start in tanh's non-saturated range.
        tn_rand_normal( layer.W, 0.0f, 1.0f / sqrtf( (float)in ) );

        nn.layers.push_back( layer );
    }
    return nn;
}

void nn_free( Network & nn ) {
    for ( Layer & layer : nn.layers ) {
        tn_free( layer.W );
        tn_free( layer.b );
        tn_free( layer.dW );
        tn_free( layer.db );
        tn_free( layer.input );
        tn_free( layer.output );
        tn_free( layer.gradInput );
    }
    nn.layers.clear();
}

void nn_zero_grad( Network & nn ) {
    for ( Layer & layer : nn.layers ) {
        tn_zero( layer.dW );
        tn_zero( layer.db );
    }
}

const Tensor & nn_forward( Network & nn, const Tensor & inputs ) {
    const Tensor * x = &inputs;
    for ( Layer & layer : nn.layers ) {
        // Cache x for the backward pass. Copy by element count so a 2D state tensor
        // can be fed into a flat input layer.
        assert( x->count == layer.input.count );
        for ( int i = 0; i < x->count; i++ ) {
            layer.input.data[i] = x->data[i];
        }

        // y = act( W x + b )
        tn_matvec( layer.output, layer.W, layer.input );
        tn_add( layer.output, layer.b );
        if ( layer.act == Activation::Tanh ) {
            for ( int i = 0; i < layer.output.count; i++ ) {
                layer.output.data[i] = tanhf( layer.output.data[i] );
            }
        }

        x = &layer.output;
    }
    return *x;
}

void nn_backward( Network & nn, const Tensor & fGrads ) {
    const Tensor * gradOutput = &fGrads;     // dLoss/dy for the current layer

    for ( int l = (int)nn.layers.size() - 1; l >= 0; l-- ) {
        Layer & layer = nn.layers[l];
        int outCount = layer.W.shape[0];
        int inCount = layer.W.shape[1];
        assert( gradOutput->count == outCount );

        tn_zero( layer.gradInput );

        for ( int o = 0; o < outCount; o++ ) {
            // dLoss/dz where z = W x + b is the pre-activation.
            // tanh'(z) = 1 - tanh(z)^2 = 1 - y^2, and y is cached in output.
            float dz = gradOutput->data[o];
            if ( layer.act == Activation::Tanh ) {
                float y = layer.output.data[o];
                dz *= 1.0f - y * y;
            }

            // z_o = sum_i W[o,i] x_i + b_o
            //   dz_o/dW[o,i] = x_i
            //   dz_o/db_o    = 1
            //   dz_o/dx_i    = W[o,i]
            tn_at( layer.db, o ) += dz;
            for ( int i = 0; i < inCount; i++ ) {
                tn_at( layer.dW, o, i ) += dz * layer.input.data[i];
                layer.gradInput.data[i] += dz * tn_at( layer.W, o, i );
            }
        }

        gradOutput = &layer.gradInput;
    }
}

void nn_update( Network & nn, float lr ) {
    for ( Layer & layer : nn.layers ) {
        tn_axpy( layer.W, -lr, layer.dW );
        tn_axpy( layer.b, -lr, layer.db );
    }
}
