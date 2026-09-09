#include "nerf_mlp.h"

#include <cmath>

namespace nerf {
    // xorshift32. Seeded per net so the same seed gives the same starting weights, and 0 is folded
    // to a non zero constant because a zero state gets stuck.
    struct RandomSeries {
        u32 state;
    };

    static u32 RandomNextU32( RandomSeries * rng ) {
        u32 x = rng->state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        rng->state = x;
        return x;
    }

    // Uniform in [ -1, 1 ].
    static f32 RandomBilateral( RandomSeries * rng ) {
        const f32 unit = (f32)( RandomNextU32( rng ) >> 8 ) * ( 1.0f / 16777216.0f );
        return unit * 2.0f - 1.0f;
    }

    static f32 ApplyActivation( ActivationFunction func, f32 x ) {
        switch( func ) {
            case ACTIVATION_FUNCTION_RELU:      return Max( 0.0f, x );
            case ACTIVATION_FUNCTION_TANH:      return tanhf( x );
            case ACTIVATION_FUNCTION_SIGMOID:   return 1.0f / ( 1.0f + expf( -x ) );
            case ACTIVATION_FUNCTION_IDENTITY:
            default:                            return x;
        }
    }

    // Derivative with respect to the pre activation. tanh and sigmoid are cheaper to write in terms
    // of their own output, which the forward pass already stored, so both values get passed in.
    static f32 ActivationDerivative( ActivationFunction func, f32 preAct, f32 postAct ) {
        switch( func ) {
            case ACTIVATION_FUNCTION_RELU:      return preAct > 0.0f ? 1.0f : 0.0f;
            case ACTIVATION_FUNCTION_TANH:      return 1.0f - postAct * postAct;
            case ACTIVATION_FUNCTION_SIGMOID:   return postAct * ( 1.0f - postAct );
            case ACTIVATION_FUNCTION_IDENTITY:
            default:                            return 1.0f;
        }
    }

    NetworkMlp * MlpCreate( const i32 * sizes, const i32 layers, ActivationFunction actHidden, ActivationFunction actOut, u32 seed ) {
        if( sizes == nullptr || layers < 2 || layers > kMlpMaxLayers ) {
            return nullptr;
        }

        for( i32 i = 0; i < layers; i++ ) {
            if( sizes[i] < 1 || sizes[i] > kMlpMaxWidth ) {
                return nullptr;
            }
        }

        NetworkMlp * mlp = new NetworkMlp();
        mlp->layerCount = layers;
        mlp->actHidden = actHidden;
        mlp->actOutput = actOut;
        for( i32 i = 0; i < layers; i++ ) {
            mlp->sizes[i] = sizes[i];
        }

        RandomSeries rng = {};
        rng.state = seed != 0 ? seed : 0x9e3779b9u;

        for( i32 l = 0; l < layers - 1; l++ ) {
            const i32 inCount = mlp->sizes[l];
            const i32 outCount = mlp->sizes[l + 1];
            const i32 weightCount = inCount * outCount;

            mlp->weights[l] = new f32[weightCount];
            mlp->biases[l] = new f32[outCount];
            mlp->weightGrads[l] = new f32[weightCount];
            mlp->biasGrads[l] = new f32[outCount];

            // He for relu, Xavier otherwise, scaled so the uniform draw has the matching variance.
            const ActivationFunction act = ( l + 2 == layers ) ? actOut : actHidden;
            const f32 gain = act == ACTIVATION_FUNCTION_RELU ? 2.0f : 1.0f;
            const f32 limit = sqrtf( 3.0f * gain / (f32)inCount );

            for( i32 i = 0; i < weightCount; i++ ) {
                mlp->weights[l][i] = RandomBilateral( &rng ) * limit;
                mlp->weightGrads[l][i] = 0.0f;
            }
            for( i32 i = 0; i < outCount; i++ ) {
                mlp->biases[l][i] = 0.0f;
                mlp->biasGrads[l][i] = 0.0f;
            }
        }

        return mlp;
    }

    void MlpDestroy( NetworkMlp * mlp ) {
        if( mlp == nullptr ) {
            return;
        }
        for( i32 l = 0; l < mlp->layerCount - 1; l++ ) {
            delete[] mlp->weights[l];
            delete[] mlp->biases[l];
            delete[] mlp->weightGrads[l];
            delete[] mlp->biasGrads[l];
            delete[] mlp->weightM[l];
            delete[] mlp->weightV[l];
            delete[] mlp->biasM[l];
            delete[] mlp->biasV[l];
        }
        delete mlp;
    }

    void MlpForward( NetworkMlp * m, const f32 * in, f32 * out ) {
        for( i32 i = 0; i < m->sizes[0]; i++ ) {
            m->activations[0][i] = in[i];
        }

        for( i32 l = 0; l < m->layerCount - 1; l++ ) {
            const i32 inCount = m->sizes[l];
            const i32 outCount = m->sizes[l + 1];
            const ActivationFunction act = ( l + 2 == m->layerCount ) ? m->actOutput : m->actHidden;
            const f32 * weights = m->weights[l];

            for( i32 o = 0; o < outCount; o++ ) {
                const f32 * row = weights + (i64)o * inCount;
                f32 sum = m->biases[l][o];
                for( i32 i = 0; i < inCount; i++ ) {
                    sum += row[i] * m->activations[l][i];
                }
                m->preActivations[l + 1][o] = sum;
                m->activations[l + 1][o] = ApplyActivation( act, sum );
            }
        }

        if( out != nullptr ) {
            const i32 last = m->layerCount - 1;
            for( i32 i = 0; i < m->sizes[last]; i++ ) {
                out[i] = m->activations[last][i];
            }
        }
    }

    void MlpBackward( NetworkMlp * m, const f32 * dLdOut ) {
        const i32 last = m->layerCount - 1;
        m->gradAccumCount++;

        for( i32 o = 0; o < m->sizes[last]; o++ ) {
            const f32 d = ActivationDerivative( m->actOutput, m->preActivations[last][o], m->activations[last][o] );
            m->deltas[last][o] = dLdOut[o] * d;
        }

        for( i32 l = last - 1; l >= 0; l-- ) {
            const i32 inCount = m->sizes[l];
            const i32 outCount = m->sizes[l + 1];
            const f32 * weights = m->weights[l];
            f32 * weightGrads = m->weightGrads[l];

            for( i32 o = 0; o < outCount; o++ ) {
                const f32 delta = m->deltas[l + 1][o];
                f32 * gradRow = weightGrads + (i64)o * inCount;
                for( i32 i = 0; i < inCount; i++ ) {
                    gradRow[i] += delta * m->activations[l][i];
                }
                m->biasGrads[l][o] += delta;
            }

            // Layer 0 holds the network input, so there is nothing further to propagate into.
            if( l == 0 ) {
                break;
            }

            for( i32 i = 0; i < inCount; i++ ) {
                f32 sum = 0.0f;
                for( i32 o = 0; o < outCount; o++ ) {
                    sum += weights[(i64)o * inCount + i] * m->deltas[l + 1][o];
                }
                m->deltas[l][i] = sum * ActivationDerivative( m->actHidden, m->preActivations[l][i], m->activations[l][i] );
            }
        }
    }

    void MlpSetOptimizerAdam( NetworkMlp * m, f32 beta1, f32 beta2, f32 eps ) {
        if( m == nullptr || m->optimizer == OPTIMIZER_ADAM ) {
            return;
        }

        m->optimizer = OPTIMIZER_ADAM;
        m->beta1 = beta1;
        m->beta2 = beta2;
        m->adamEps = eps;
        m->beta1Pow = 1.0f;
        m->beta2Pow = 1.0f;

        for( i32 l = 0; l < m->layerCount - 1; l++ ) {
            const i32 outCount = m->sizes[l + 1];
            const i32 weightCount = m->sizes[l] * outCount;

            m->weightM[l] = new f32[weightCount]();
            m->weightV[l] = new f32[weightCount]();
            m->biasM[l] = new f32[outCount]();
            m->biasV[l] = new f32[outCount]();
        }
    }

    // One Adam parameter update. The caller folds bias correction into the step size and passes the
    // already averaged gradient, so this stays a straight read-modify-write.
    static void AdamStep( f32 * param, f32 * moment1, f32 * moment2, const f32 * grads, i32 count, f32 lr, f32 beta1, f32 beta2, f32 eps, f32 bias1, f32 bias2 ) {
        for( i32 i = 0; i < count; i++ ) {
            const f32 g = grads[i];
            moment1[i] = beta1 * moment1[i] + ( 1.0f - beta1 ) * g;
            moment2[i] = beta2 * moment2[i] + ( 1.0f - beta2 ) * g * g;

            const f32 mHat = moment1[i] / bias1;
            const f32 vHat = moment2[i] / bias2;
            param[i] -= lr * mHat / ( sqrtf( vHat ) + eps );
        }
    }

    void MlpApplyGrads( NetworkMlp * m, f32 lr ) {
        // Nothing accumulated means no step to take, and it keeps the averaging below off zero.
        if( m->gradAccumCount == 0 ) {
            return;
        }

        // Gradients are summed over the mini batch, so average them here. Without this the
        // effective learning rate would scale with however many samples went into the batch.
        const f32 invBatch = 1.0f / (f32)m->gradAccumCount;

        f32 bias1 = 1.0f;
        f32 bias2 = 1.0f;
        if( m->optimizer == OPTIMIZER_ADAM ) {
            // Tracked incrementally rather than with powf( beta, t ): same value, and it cannot
            // overflow the step counter on a long training run.
            m->beta1Pow *= m->beta1;
            m->beta2Pow *= m->beta2;
            bias1 = 1.0f - m->beta1Pow;
            bias2 = 1.0f - m->beta2Pow;
        }

        for( i32 l = 0; l < m->layerCount - 1; l++ ) {
            const i32 inCount = m->sizes[l];
            const i32 outCount = m->sizes[l + 1];
            const i32 weightCount = inCount * outCount;

            for( i32 i = 0; i < weightCount; i++ ) {
                m->weightGrads[l][i] *= invBatch;
            }
            for( i32 o = 0; o < outCount; o++ ) {
                m->biasGrads[l][o] *= invBatch;
            }

            if( m->optimizer == OPTIMIZER_ADAM ) {
                AdamStep( m->weights[l], m->weightM[l], m->weightV[l], m->weightGrads[l], weightCount, lr, m->beta1, m->beta2, m->adamEps, bias1, bias2 );
                AdamStep( m->biases[l], m->biasM[l], m->biasV[l], m->biasGrads[l], outCount, lr, m->beta1, m->beta2, m->adamEps, bias1, bias2 );
            } else {
                for( i32 i = 0; i < weightCount; i++ ) {
                    m->weights[l][i] -= lr * m->weightGrads[l][i];
                }
                for( i32 o = 0; o < outCount; o++ ) {
                    m->biases[l][o] -= lr * m->biasGrads[l][o];
                }
            }
        }

        MlpZeroGrads( m );
    }

    void MlpZeroGrads( NetworkMlp * m ) {
        m->gradAccumCount = 0;
        for( i32 l = 0; l < m->layerCount - 1; l++ ) {
            const i32 inCount = m->sizes[l];
            const i32 outCount = m->sizes[l + 1];
            const i32 weightCount = inCount * outCount;

            for( i32 i = 0; i < weightCount; i++ ) {
                m->weightGrads[l][i] = 0.0f;
            }
            for( i32 o = 0; o < outCount; o++ ) {
                m->biasGrads[l][o] = 0.0f;
            }
        }
    }
}
