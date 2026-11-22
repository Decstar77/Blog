#pragma once
#include "CSGame.h"
#include "ProximalPolicyOptim.h"

class CSPPO : public PPOInterface {
  public:
    virtual void    new_game() override;
    virtual bool    is_terminal() override;
    virtual void    step( PlayerAction action ) override;
    virtual float   last_reward() override;

    virtual Network create_actor() override;
    virtual Network create_critic() override;

  private:
    int     winner;
    CSGame  cs;
};
