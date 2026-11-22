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
    return Network();
}




