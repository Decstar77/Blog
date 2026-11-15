#include "OneStepActorCritic.h"

ActorCritResult one_step_actor_critic( int episodeCount, float alphaTheta, float alphaW, float gamma, ActorCriticInterface * ac ) {
    Tensor theta = ac->intial_theta();
    Tensor w = ac->inital_w();
    for ( int episodeNumber = 0; episodeNumber < episodeCount; episodeNumber++ ) {
        int s = ac->new_game();
        int I = 1;
        float r = 0;
        while ( ac->is_terminal() == false ) {
            int a = ac->next_action( s, theta );
            int sp = ac->step( a, r );

            float delta = r + gamma * ac->state_value( sp, w ) - ac->state_value( s, w );

            Tensor wGrad = ac->state_value_grad( s, w );
            tn_mul( alphaW * delta, wGrad );
            tn_add( w, wGrad );

            Tensor thetaGrad = ac->theta_log_grad( s, theta );
            tn_mul( alphaTheta * I * delta, thetaGrad );
            tn_add( theta, thetaGrad );

            I = gamma * I;
            s = sp;
        }
    }

    ActorCritResult res;
    res.actor = theta;
    res.critic = w;
    return res;
}