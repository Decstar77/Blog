#pragma once

#include "Tensor.h"
#include "Network.h"
#include "CSGame.h"    // PlayerAction

/*
===============================================================================
===============================================================================
*/
class PPOInterface {
  public:
    virtual void            new_game() = 0;
    virtual bool            is_terminal() = 0;
    virtual void            step( PlayerAction action ) = 0;
    virtual float           last_reward() = 0;

    // Returned tensor is owned by the caller (tn_free it).
    virtual Tensor          get_state_tensor() = 0;

    // Draw an action from pi( . | actorOutput ).
    virtual PlayerAction    sample_action( const Tensor & actorOutput ) = 0;

     // log pi( action | actorOutput )
    virtual float           log_prob( const Tensor & actorOutput, const PlayerAction & action ) = 0;

    // grad = d log pi( action ) / d actorOutput.        Softmax logits: one_hot(a) - pi
    virtual void            log_prob_grad( const Tensor & actorOutput, const PlayerAction & action, Tensor & grad ) = 0;

    // grad = d H( pi ) / d actorOutput.                  Softmax logits: -pi_j * ( log pi_j + H )
    virtual void            entropy_grad( const Tensor & actorOutput, Tensor & grad ) = 0;

    // Actor's output feeds log_prob etc. Critic must output a single value.
    virtual Network         create_actor() = 0;
    virtual Network         create_critic() = 0;
};

/*
===============================================================================
===============================================================================
*/
struct PPOSettings {
    float   gamma;
    float   lambda;
    float   clipping;
    float   entopy;
    int     rollout;
    int     epochs;
    int     batchSize;
    float   alphaActor;
    float   alphaCritic;
    int     iterationCount;
};

struct PPOResult {
    Network actor;
    Network critic;
};

PPOResult ppo( PPOSettings settings, PPOInterface * model );
