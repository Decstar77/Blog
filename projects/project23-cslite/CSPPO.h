#pragma once
#include "CSGame.h"
#include "ProximalPolicyOptim.h"
#include "Network.h"
#include "Util.h"


class CSPPO : public PPOInterface {
  public:
    virtual void    new_game() override;
    virtual bool    is_terminal() override;
    virtual void    step( PlayerAction action ) override;
    virtual float   last_reward() override;

    virtual Network create_actor() override;
    virtual Network create_critic() override;

    virtual Tensor          get_state_tensor() override;
    virtual PlayerAction    sample_action( const Tensor & actorOutput ) override ;
    virtual float           log_prob( const Tensor & actorOutput, const PlayerAction & action ) override; // log pi( action | actorOutput )
    // grad = d log pi( action ) / d actorOutput.        Softmax logits: one_hot(a) - pi
    virtual void log_prob_grad( const Tensor & actorOutput, const PlayerAction & action, Tensor & grad ) = 0;

    // grad = d H( pi ) / d actorOutput.                  Softmax logits: -pi_j * ( log pi_j + H )
    virtual void entropy_grad( const Tensor & actorOutput, Tensor & grad ) = 0;

  private:
    int     winner;
    CSGame  cs;

    inline static int ACTOR_OUTPUT_SIZE = 4 + 1 + 1;
};
