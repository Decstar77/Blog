#include "ProximalPolicyOptim.h"

#include <vector>

struct Step {
    float vCurr;
    float vNext;
    float reward;
    float logprob;

    float delta;
    float ahat;
    float retn;

    Tensor state;
};

inline int min( int a, int b ) {
    return a < b ? a : b;
}

inline float clamp(float x, float min, float max) {
    if ( x < min ) {
        x = min;
    }
    if (x > max) {
        x = max;
    }
    return x;
}

inline std::vector<int> Shuffle( int size ) {
}

void ppo( PPOSettings settings, PPOInterface * model ) {
    Network actor = model->create_actor();
    Network critic = model->create_critic();

    for ( int iteration = 0; iteration < settings.iterationCount; iteration++ ) {
        std::vector<Step> steps;
        for ( int t = 0; t < settings.rollout; t++ ) {
            Tensor state = model->get_state_tensor();
            Tensor actionTensor = nn_forward( actor, state );
            PlayerAction action = model->action_tensor_to_action( actionTensor );
            model->step( action );
            Tensor nextState = model->get_state_tensor();
            Tensor v = nn_forward( critic, state );

            Step st = steps.emplace_back();
            st.logprob = model->log_prob( actionTensor );
            st.state = state;
            st.reward = model->last_reward();
            st.vCurr = tn_value( v );
            st.vNext = 0;
            if ( model->is_terminal() == false ) {
                Tensor vn = nn_forward( critic, state );
                st.vNext = tn_value( v );
            } else {
                break;
            }
        }

        const int T = (int) steps.size();
        float ahat = 0;
        for ( int t = T - 1; t >= 0; t-- ) {
            Step & step = steps[t];
            double delta = step.reward + settings.gamma * step.vNext - step.vCurr;
            ahat = delta + settings.gamma * settings.lambda * ahat;
            step.ahat = ahat;
            step.retn = ahat + step.vCurr;
        }

        const int T = (int) steps.size();
        float mean = 0;
        for ( int t = 0; t < T; t++ ) {
            Step & step = steps[t];
            mean += step.ahat;
        }
        mean /= T;

        float std = 0;
        for ( int t = 0; t < T; t++ ) {
            Step & step = steps[t];
            float d = step.ahat - mean;
            d *= d;
            std += d;
        }
        std /= T;
        std = sqrtf( std );

        for ( int t = 0; t < T; t++ ) {
            Step & step = steps[t];
            step.ahat = ( step.ahat - mean ) / std;
        }

        for ( int epoch = 0; epoch < settings.epochs; epoch++ ) {
            std::vector<int> order = Shuffle( T );

            for ( int start = 0; start < T; start += settings.batchSize ) {
                int end = min( start + settings.batchSize, T );
                float M = 1.0 / ( end - start );
                nn_zero_grad( actor );
                nn_zero_grad( critic );

                for ( int k = start; k < end; k++ ) {
                    Step step = steps[order[k]];
                    Tensor at = nn_forward( actor, step.state );
                    float ratio = exp( model->log_prob( at ) + step.logprob );
                    float lclip = min( ratio * step.ahat, clamp( ratio, 1 - settings.clipping, 1 + settings.clipping ) * step.ahat );
                    float entropy = model->entropy( at );

                    Tensor fActor;
                    nn_backward( actor, fActor );

                }

                nn_update( actor, settings.alphaActor * M );
                nn_update( critic, settings.alphaCritic * M );
            }
        }
    }
}
