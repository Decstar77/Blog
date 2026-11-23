#include "CSGame.h"

#include "raylib.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "CSPPO.h"

/*
==========================
==========================
*/
static PlayerAction dummy_action( const CSGame & cs ) {
    const Player & self = cs.players[ 1 ];
    const Player & target = cs.players[ 0 ];
    PlayerAction a = {};
    a.rotation = cs_turn_toward( self.rotation, atan2f( target.pos.y - self.pos.y, target.pos.x - self.pos.x ) );
    return a;
}

/*
==========================
==========================
*/
static PlayerAction actor_action( CSPPO & csppo, PPOResult & result ) {
    Tensor state = csppo.get_state_tensor();
    Tensor action = nn_forward( result.actor, state );
    PlayerAction p2 = csppo.sample_action( action );
    tn_free( state );
    return p2;
}

/*
==========================
==========================
 */
int main() {
    PPOSettings settings = {};
    settings.gamma = 0.990f;
    settings.lambda = 0.95f;
    settings.clipping = 0.2f;
    settings.entopy = 0.025f;
    settings.rollout = 8192;
    settings.epochs = 10;
    settings.batchSize = 128;
    settings.alphaActor = 0.01f;
    settings.alphaCritic = 0.01f;
    settings.iterationCount = 200;

    CSPPO csppo;
    PPOResult result = ppo( settings, &csppo );

    CSGame & cs = csppo.cs;
    cs_new_game( cs );

    InitWindow( CS_MAP_WIDTH * CS_TILE_PIXELS, CS_MAP_HEIGHT * CS_TILE_PIXELS, "CS Lite" );
    SetTargetFPS( 144 );

    // Fixed-step simulation, render as fast as allowed
    float accumulator = 0.0f;
    while ( !WindowShouldClose() ) {
        if ( IsKeyPressed( KEY_R ) ) {
            cs_new_game( cs );
        }

        accumulator += fminf( GetFrameTime(), 0.25f );
        while ( accumulator >= CS_DT ) {
            PlayerAction p1 = cs_input_from_keyboard( cs, 0 );
            //PlayerAction p2 =  dummy_action( cs );
            PlayerAction p2 = actor_action( csppo, result );
            cs_step( cs, p1, p2 );
            accumulator -= CS_DT;
        }

        BeginDrawing();
        cs_render( cs );
        EndDrawing();
    }

    CloseWindow();
    cs_free( cs );
    return 0;
}
