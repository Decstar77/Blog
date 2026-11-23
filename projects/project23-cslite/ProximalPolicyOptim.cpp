#include "ProximalPolicyOptim.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

struct Step {
    Tensor          state;      // owned, freed at the end of the iteration
    PlayerAction    action;
    float           logprob;    // log pi_old( action | state ), fixed during the epochs
    float           reward;
    float           vCurr;      // V( s_t )
    float           vNext;      // V( s_t+1 ), 0 if s_t+1 is terminal
    bool            done;       // episode ended after this step

    float           ahat;       // GAE advantage
    float           retn;       // critic target: ahat + vCurr
};

/*
==========================
==========================
 */
inline int min( int a, int b ) {
    return a < b ? a : b;
}

/*
==========================
==========================
 */
inline std::vector<int> Shuffle( int size ) {
    static std::mt19937 rng( 1234 );
    std::vector<int> order( size );
    for ( int i = 0; i < size; i++ ) {
        order[i] = i;
    }
    std::shuffle( order.begin(), order.end(), rng );
    return order;
}

/*
==========================
==========================
 */
PPOResult ppo( PPOSettings settings, PPOInterface * model ) {
    Network actor = model->create_actor();
    Network critic = model->create_critic();

    model->new_game();

    // Episodes can span rollout boundaries, so the running episode totals live outside the loop.
    float episodeReward = 0.0f;
    int   episodeLength = 0;

    printf( "%5s | %5s %8s %8s | %8s %8s %8s %8s %8s\n",
            "iter", "eps", "ep_rew", "ep_len", "value", "v_loss", "surr", "kl", "clip%" );

    for ( int iteration = 0; iteration < settings.iterationCount; iteration++ ) {
        // Rollout stats, reset every iteration
        int   episodesFinished = 0;
        float sumEpisodeReward = 0.0f;
        int   sumEpisodeLength = 0;

        // ---- Rollout -----------------------------------------------------------
        std::vector<Step> steps;
        for ( int t = 0; t < settings.rollout; t++ ) {
            Step & st = steps.emplace_back();
            st.state = model->get_state_tensor();

            const Tensor & actorOut = nn_forward( actor, st.state );
            st.action = model->sample_action( actorOut );
            st.logprob = model->log_prob( actorOut, st.action );

            st.vCurr = tn_value( nn_forward( critic, st.state ) );

            model->step( st.action );
            st.reward = model->last_reward();
            st.done = model->is_terminal();

            episodeReward += st.reward;
            episodeLength += 1;

            if ( st.done ) {
                episodesFinished += 1;
                sumEpisodeReward += episodeReward;
                sumEpisodeLength += episodeLength;
                episodeReward = 0.0f;
                episodeLength = 0;

                st.vNext = 0.0f;
                model->new_game();
            } else {
                Tensor nextState = model->get_state_tensor();
                st.vNext = tn_value( nn_forward( critic, nextState ) );
                tn_free( nextState );
            }
        }

        const int T = (int) steps.size();

        // ---- GAE --------------------------------------------------------------------------
        float ahat = 0;
        for ( int t = T - 1; t >= 0; t-- ) {
            Step & step = steps[t];
            if ( step.done ) {
                ahat = 0;
            }
            float delta = step.reward + settings.gamma * step.vNext - step.vCurr;
            ahat = delta + settings.gamma * settings.lambda * ahat;
            step.ahat = ahat;
            step.retn = ahat + step.vCurr;
        }

        // ---- Normalize advantages to mean 0, std 1 ----------------------------------------
        float mean = 0;
        for ( int t = 0; t < T; t++ ) {
            mean += steps[t].ahat;
        }
        mean /= T;

        float std = 0;
        for ( int t = 0; t < T; t++ ) {
            float d = steps[t].ahat - mean;
            std += d * d;
        }
        std = sqrtf( std / T );

        for ( int t = 0; t < T; t++ ) {
            steps[t].ahat = ( steps[t].ahat - mean ) / ( std + 1e-8f );
        }

        // Optimization stats, averaged over every sample of every epoch
        float sumCriticLoss = 0.0f;   // ( V(s) - retn )^2
        float sumSurrogate  = 0.0f;   // min( r * A, clip( r ) * A ), the thing the actor maximizes
        float sumApproxKL   = 0.0f;   // log pi_old - log pi_new, how far the policy moved
        int   clippedCount  = 0;      // samples where the clip zeroed the actor gradient
        int   sampleCount   = 0;

        // ---- Optimize: several epochs of mini-batch updates over the same rollout ---------
        for ( int epoch = 0; epoch < settings.epochs; epoch++ ) {
            std::vector<int> order = Shuffle( T );

            for ( int start = 0; start < T; start += settings.batchSize ) {
                int end = min( start + settings.batchSize, T );
                float M = 1.0f / ( end - start );
                nn_zero_grad( actor );
                nn_zero_grad( critic );

                for ( int k = start; k < end; k++ ) {
                    Step & step = steps[order[k]];

                    // -- Actor --------------------------------------------------------------
                    const Tensor & out = nn_forward( actor, step.state );
                    float newLogprob = model->log_prob( out, step.action );
                    float ratio = expf( newLogprob - step.logprob );

                    // dL/dr: the min picks the clipped term (gradient 0) exactly when the policy
                    bool clipped = ( step.ahat > 0 && ratio > 1 + settings.clipping ) || ( step.ahat < 0 && ratio < 1 - settings.clipping );
                    float dLdr = clipped ? 0.0f : step.ahat;

                    float clippedRatio = std::clamp( ratio, 1 - settings.clipping, 1 + settings.clipping );
                    sumSurrogate += std::min( ratio * step.ahat, clippedRatio * step.ahat );
                    sumApproxKL  += step.logprob - newLogprob;
                    clippedCount += clipped ? 1 : 0;
                    sampleCount  += 1;

                    // dr/dlogpi = r, so dL/dlogpi = dLdr * r
                    float dLdLogp = dLdr * ratio;

                    // dL/dout = dL/dlogpi * dlogpi/dout + c * dH/dout
                    Tensor gradLogp = tn_create_shape( out.dim, out.shape );
                    Tensor gradH    = tn_create_shape( out.dim, out.shape );

                    model->log_prob_grad( out, step.action, gradLogp );
                    model->entropy_grad( out, gradH );

                    Tensor gradOut = tn_create_shape( out.dim, out.shape );
                    tn_axpy( gradOut, dLdLogp, gradLogp );
                    tn_axpy( gradOut, settings.entopy, gradH );

                    // nn_update descends, so feed it the gradient of the loss -L.
                    tn_mul( -1.0f, gradOut );
                    nn_backward( actor, gradOut );

                    tn_free( gradLogp );
                    tn_free( gradH );
                    tn_free( gradOut );

                    // -- Critic -------------------------------------------------------------
                    // Loss = ( V(s) - retn )^2,  dLoss/dV = 2 ( V(s) - retn )
                    float v = tn_value( nn_forward( critic, step.state ) );
                    sumCriticLoss += ( v - step.retn ) * ( v - step.retn );

                    Tensor gradV = tn_create( 1 );
                    tn_at( gradV, 0 ) = 2.0f * ( v - step.retn );
                    nn_backward( critic, gradV );
                    tn_free( gradV );
                }

                // Grads were summed over the batch; M = 1 / batch size turns that into the mean.
                nn_update( actor, settings.alphaActor * M );
                nn_update( critic, settings.alphaCritic * M );
            }
        }

        // ---- Log ---------------------------------------------------------------------------
        if ( iteration % 100 == 0 ) {

            float meanValue = 0.0f;
            for ( int t = 0; t < T; t++ ) {
                meanValue += steps[t].vCurr;
            }
            meanValue /= T;

            // No episode finished this rollout: print the one still running so the columns aren't empty.
            float meanEpisodeReward = episodesFinished > 0 ? sumEpisodeReward / episodesFinished : episodeReward;
            float meanEpisodeLength = episodesFinished > 0 ? (float) sumEpisodeLength / episodesFinished : (float) episodeLength;

            printf( "%5d | %5d %8.3f %8.1f | %8.3f %8.4f %8.4f %8.5f %7.1f%%\n",
                iteration,
                episodesFinished,
                meanEpisodeReward,
                meanEpisodeLength,
                meanValue,
                sumCriticLoss / sampleCount,
                sumSurrogate / sampleCount,
                sumApproxKL / sampleCount,
                100.0f * clippedCount / sampleCount );
            fflush( stdout );
        }

        // ---- Sss ---------------------------------------------------------------------------
        for ( Step & step : steps ) {
            tn_free( step.state );
        }
    }

    PPOResult res;
    res.actor = actor;
    res.critic = critic;
    return res;
}
