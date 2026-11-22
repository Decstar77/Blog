#include "Tensor.h"
#include "Network.h"

struct PlayerAction {

};

class PPOInterface {
  public:
    virtual int     new_game() = 0;
    virtual bool    is_terminal() = 0;
    virtual void    step( PlayerAction action ) = 0;

    virtual Tensor          get_state_tensor() = 0;
    virtual PlayerAction    action_tensor_to_action( const Tensor & action ) = 0;
    virtual float           last_reward() = 0;
    virtual float           log_prob( const Tensor & action ) = 0;
    virtual float           entropy( const Tensor & action ) = 0;

    virtual Network create_actor() = 0;
    virtual Network create_critic() = 0;
};

struct PPOSettings {
    float   gamma;
    float   lambda;
    float   clipping;
    float   entopy;
    int     trajectories;
    int     rollout;
    int     epochs;
    int     batchSize;
    float   alphaActor;
    float   alphaCritic;
    int     iterationCount;
};

void ppo( PPOSettings settings, PPOInterface * model );

