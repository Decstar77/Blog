#include "CSPPO.h"

void CSPPO::new_game() {
    cs_new_game( cs );
}

bool CSPPO::is_terminal() {
    cs_is_terminal( cs, winner );
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
    return nn_create(sizes);
}

Network CSPPO::create_critic() {
    Tensor dummy = get_state_tensor();
    std::vector<int> sizes = { dummy.shape[0], 1 };
    return nn_create( sizes );
}

Tensor CSPPO::get_state_tensor() {
    // TODO

}

PlayerAction CSPPO::sample_action( const Tensor & actorOutput ) {
    float m1 = ( actorOutput.data[0] + 1 ) / 2;
    float m2 = ( actorOutput.data[1] + 1 ) / 2;
    float m3 = ( actorOutput.data[2] + 1 ) / 2;
    float m4 = ( actorOutput.data[3] + 1 ) / 2;
    float f = ( actorOutput.data[4] + 1 ) / 2;
    float r = actorOutput.data[6];

    m1 = logf( m1 / ( 1 - m1 ) );
    m2 = logf( m2 / ( 1 - m2 ) );
    m3 = logf( m3 / ( 1 - m3 ) );
    m4 = logf( m4 / ( 1 - m4 ) );
    f = logf( m1 / ( 1 - m1 ) );

    PlayerAction action = {};
    action.moveDown  = random_normal() < m1;
    action.moveUp    = random_normal() < m2;
    action.moveLeft  = random_normal() < m3;
    action.moveRight = random_normal() < m4;
    action.shoot     = random_normal() < f;
    action.rotation  = r;

    return action;
}

float CSPPO::log_prob( const Tensor & actorOutput, const PlayerAction & action ) {
    float p = 0;
    if ( action.moveDown ) { p += logf( actorOutput.data[0] ); }
    if ( action.moveUp ) { p += logf( actorOutput.data[1] ); }
    if ( action.moveLeft ) { p += logf( actorOutput.data[2] ); }
    if ( action.moveRight ) { p += logf( actorOutput.data[3] ); }
    if ( action.shoot ) { p += logf( action.shoot ); }
    return p;
}






