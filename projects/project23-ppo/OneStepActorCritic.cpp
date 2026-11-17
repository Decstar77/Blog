#include "OneStepActorCritic.h"

ActorCritResult one_step_actor_critic( int episodeCount, float alphaTheta, float alphaW, float gamma, ActorCriticInterface * ac ) {
    Tensor theta = ac->initial_theta();
    Tensor w = ac->initial_w();
    for ( int episodeNumber = 0; episodeNumber < episodeCount; episodeNumber++ ) {
        int s = ac->new_game();
        float I = 1.0f;
        while ( ac->is_terminal() == false ) {
            int a = ac->next_action( s, theta );
            float r = 0;
            int sp = ac->step( a, r );

            float vNext = ac->is_terminal() ? 0.0f : ac->state_value( sp, w );
            float delta = r + gamma * vNext - ac->state_value( s, w );

            Tensor wGrad = ac->state_value_grad( s, w );
            tn_mul( alphaW * delta, wGrad );
            tn_add( w, wGrad );

            Tensor thetaGrad = ac->theta_log_grad( s, a, theta );
            tn_mul( alphaTheta * I * delta, thetaGrad );
            tn_add( theta, thetaGrad );

            I *= gamma;
            s = sp;
        }
    }

    ActorCritResult res;
    res.actor = theta;
    res.critic = w;
    return res;
}

ActorCritResult one_step_actor_critic_etraces( ActorCriticSettings settings, ActorCriticInterface * ac ) {
    Tensor theta = ac->initial_theta();
    Tensor w = ac->initial_w();

    Tensor zTheta = tn_clone( theta );
    Tensor zW = tn_clone( w );

    Tensor zThetaTemp = tn_clone( theta );
    Tensor zWTemp = tn_clone( w );

    for ( int episodeNumber = 0; episodeNumber < settings.episodeCount; episodeNumber++ ) {
        int s = ac->new_game();

        tn_zero( zTheta );
        tn_zero( zW );
        tn_zero( zThetaTemp );
        tn_zero( zWTemp );

        float I = 1.0f;
        while ( ac->is_terminal() == false ) {
            int a = ac->next_action( s, theta );
            float r = 0;
            int sp = ac->step( a, r );

            float vNext = ac->is_terminal() ? 0.0f : ac->state_value( sp, w );
            float delta = r + settings.gamma * vNext - ac->state_value( s, w );

            Tensor wGrad = ac->state_value_grad( s, w );
            tn_mul( settings.gamma * settings.lamdaW, zW );
            tn_add( zW, wGrad );

            Tensor thetaGrad = ac->theta_log_grad( s, a, theta );
            tn_mul( I, thetaGrad );
            tn_mul( settings.gamma * settings.lamdaTheta, zTheta );
            tn_add( zTheta, thetaGrad );

            zThetaTemp =

            I *= settings.gamma;
            s = sp;
        }
    }

    ActorCritResult res;
    res.actor = theta;
    res.critic = w;
    return res;
}
