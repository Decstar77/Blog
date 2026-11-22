#include "CSPPO.h"
#include <cmath>

void CSPPO::new_game() {
    cs_new_game( cs );
}

bool CSPPO::is_terminal() {
    return cs_is_terminal( cs, winner );
}

void CSPPO::step( PlayerAction action ) {
    cs_step( cs, action, {} );
}

float CSPPO::last_reward() {
    if ( winner == 0 ) {
        return 1.0f;
    }
    return 0.0f;
}

Network CSPPO::create_actor() {
    Tensor dummy = get_state_tensor();
    std::vector<int> sizes = { dummy.shape[0], ACTOR_OUTPUT_SIZE };
    tn_free( dummy );
    return nn_create(sizes);
}

Network CSPPO::create_critic() {
    Tensor dummy = get_state_tensor();
    std::vector<int> sizes = { dummy.shape[0], 1 };
    tn_free( dummy );
    return nn_create( sizes );
}

Tensor CSPPO::get_state_tensor() {
    // The agent is always player 0, the enemy is player 1.
    const Player & me    = cs.players[ 0 ];
    const Player & enemy = cs.players[ 1 ];

    // Positions scaled from map units into [-1, 1] so they sit in tanh's useful range.
    // Rotations as cos/sin so the angle has no jump where it wraps around.
    // Enemy position is relative to me; the difference spans [-W, W], so divide by W for [-1, 1].
    Tensor state = tn_create( 8 );
    state.data[0] = me.pos.x / CS_MAP_WIDTH  * 2.0f - 1.0f;
    state.data[1] = me.pos.y / CS_MAP_HEIGHT * 2.0f - 1.0f;
    state.data[2] = ( enemy.pos.x - me.pos.x ) / CS_MAP_WIDTH;
    state.data[3] = ( enemy.pos.y - me.pos.y ) / CS_MAP_HEIGHT;
    state.data[4] = cosf( me.rotation );
    state.data[5] = sinf( me.rotation );
    state.data[6] = cosf( enemy.rotation );
    state.data[7] = sinf( enemy.rotation );
    return state;
}

static const float ROT_STD = 0.3f;
static const float LOG_SQRT_2PI = 0.9189385332f; // 0.5 * log( 2 * pi )

static float sigmoid( float z ) { 
    return 1.0f / ( 1.0f + expf( -z ) ); 
}

static float log_sigmoid( float z ) {
      if ( z >= 0 ) {
        return -log1pf( expf( -z ) );
    }
    return z - log1pf( expf( z ) );
}

static float bernoulli_log_prob( float logit, bool taken ) {
    return taken ? log_sigmoid( logit ) : log_sigmoid( -logit );
}

static void get_bernoulli_actions( const PlayerAction & action, bool taken[5] ) {
    taken[0] = action.moveDown;
    taken[1] = action.moveUp;
    taken[2] = action.moveLeft;
    taken[3] = action.moveRight;
    taken[4] = action.shoot;
}

PlayerAction CSPPO::sample_action( const Tensor & actorOutput ) {
    PlayerAction action = {};
    action.moveDown  = random_uniform() < sigmoid( actorOutput.data[0] );
    action.moveUp    = random_uniform() < sigmoid( actorOutput.data[1] );
    action.moveLeft  = random_uniform() < sigmoid( actorOutput.data[2] );
    action.moveRight = random_uniform() < sigmoid( actorOutput.data[3] );
    action.shoot     = random_uniform() < sigmoid( actorOutput.data[4] );

    float mu = actorOutput.data[5];
    action.rotation = mu + ROT_STD * random_normal();

    return action;
}

float CSPPO::log_prob( const Tensor & actorOutput, const PlayerAction & action ) {
    bool taken[5];
    get_bernoulli_actions( action, taken );

    float lp = 0.0f;
    for ( int i = 0; i < 5; i++ ) {
        lp += bernoulli_log_prob( actorOutput.data[i], taken[i] );
    }

    // Gaussian: log N( r | mu, std ) = -0.5 * ( ( r - mu ) / std )^2 - log( std ) - 0.5 * log( 2 pi )
    float mu = actorOutput.data[5];
    float d = ( action.rotation - mu ) / ROT_STD;
    lp += -0.5f * d * d - logf( ROT_STD ) - LOG_SQRT_2PI;

    return lp;
}

void CSPPO::log_prob_grad( const Tensor & actorOutput, const PlayerAction & action, Tensor & grad ) {
    bool taken[5];
    get_bernoulli_actions( action, taken );

    // Bernoulli with logit z: d log p( a ) / dz = a - sigmoid( z )
    for ( int i = 0; i < 5; i++ ) {
        float a = taken[i] ? 1.0f : 0.0f;
        grad.data[i] = a - sigmoid( actorOutput.data[i] );
    }

    // Gaussian with mean mu: d log N / d mu = ( r - mu ) / std^2
    float mu = actorOutput.data[5];
    grad.data[5] = ( action.rotation - mu ) / ( ROT_STD * ROT_STD );
}

void CSPPO::entropy_grad( const Tensor & actorOutput, Tensor & grad ) {
    // Bernoulli with logit z, p = sigmoid( z ):
    //   H     = -p log p - ( 1 - p ) log( 1 - p )
    //   dH/dp = log( ( 1 - p ) / p ) = -z
    //   dp/dz = p ( 1 - p )
    //   dH/dz = -z * p * ( 1 - p )
    for ( int i = 0; i < 5; i++ ) {
        float z = actorOutput.data[i];
        float p = sigmoid( z );
        grad.data[i] = -z * p * ( 1.0f - p );
    }

    // Gaussian entropy depends only on std, which is fixed, so it has no gradient w.r.t. the mean.
    grad.data[5] = 0.0f;
}
