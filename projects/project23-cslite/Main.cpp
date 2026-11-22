#include "CSGame.h"

#include "raylib.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

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
int main() {
    CSGame cs = {};
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
            PlayerAction p2 = dummy_action( cs );
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
