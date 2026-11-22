#pragma once
#include "MemoryArena.h"

/*
===============================================================================
===============================================================================
*/
struct PlayerAction {
    bool moveLeft;
    bool moveRight;
    bool moveUp;
    bool moveDown;
    bool shoot;
    float rotation;
};

struct Player {
    int playerNumber;
    int health;
};

struct Map {

};

struct CSGame {

};

void cs_new_game( CSGame & cs );
void cs_step( CSGame & cs, PlayerAction p1, PlayerAction p2 );
void cs_is_terminal( int winner );
void cs_render( CSGame & cs, float dt );

