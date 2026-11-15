#pragma once

#include "Tensor.h"

class ActorCriticInterface {
  public:
    virtual int     new_game() = 0;
    virtual bool    is_terminal() = 0;
    virtual int     step( int action, float & reward ) = 0;

    virtual int     next_action( int state, const Tensor & theta ) = 0;
    virtual float   state_value( int state, const Tensor & w ) = 0;

    virtual Tensor  initial_theta() = 0;
    virtual Tensor  initial_w() = 0;

    // Gradient of ln pi(action | state, theta) with respect to theta
    virtual Tensor  theta_log_grad( int state, int action, const Tensor & theta ) = 0;
    // Gradient of v(state, w) with respect to w
    virtual Tensor  state_value_grad( int state, const Tensor & w ) = 0;
};

struct ActorCritResult {
    Tensor actor;
    Tensor critic;
};

ActorCritResult one_step_actor_critic( int episodeCount, float alphaTheta, float alphaW, float gamma, ActorCriticInterface * ac );
