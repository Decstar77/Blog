#pragma once
#include "MemoryArena.h"

#include <cstdint>

/*
===============================================================================
Two-player top-down shooter. The simulation is fixed-step and has no
dependency on raylib, so it can run headless for training. Rendering and
keyboard/mouse input live in the cs_render / cs_input_from_keyboard section.

World units: 1 unit = 1 map tile. +x right, +y down (screen convention).
===============================================================================
*/

constexpr float CS_DT               = 1.0f / 60.0f;    // seconds per cs_step
constexpr int   CS_MAP_WIDTH        = 32;               // tiles
constexpr int   CS_MAP_HEIGHT       = 24;               // tiles
constexpr int   CS_MAX_BULLETS      = 128;
constexpr int   CS_MAX_TICKS        = 60 * 60;          // 60s round, then draw

constexpr float CS_PLAYER_RADIUS    = 0.35f;
constexpr float CS_PLAYER_SPEED     = 5.0f;             // units/s
constexpr int   CS_PLAYER_HEALTH    = 100;
constexpr float CS_SHOOT_COOLDOWN   = 0.25f;            // s
constexpr float CS_BULLET_SPEED     = 18.0f;            // units/s
constexpr float CS_BULLET_RADIUS    = 0.08f;
constexpr int   CS_BULLET_DAMAGE    = 20;

struct Vec2 {
    float x;
    float y;
};

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
    float rotation;     // absolute facing angle in radians, 0 = +x
};

struct Player {
    int playerNumber;   // 0 or 1
    int health;
    Vec2 pos;
    float rotation;
    float shootCooldown;
};

struct Bullet {
    Vec2 pos;
    Vec2 vel;
    int owner;          // playerNumber of the shooter
};

enum class Tile : uint8_t {
    Empty,
    Wall,
};

struct Map {
    int width;
    int height;
    Tile * tiles;       // width * height, row-major, allocated from the game arena
};

struct CSGame {
    Arena arena;        // owns map tiles and bullets; reset by cs_new_game
    Map map;
    Player players[ 2 ];
    Bullet * bullets;   // CS_MAX_BULLETS slots
    int bulletCount;
    int tick;
};

void    cs_new_game( CSGame & cs );
void    cs_free( CSGame & cs );
void    cs_step( CSGame & cs, PlayerAction p1, PlayerAction p2 );
bool    cs_is_terminal( const CSGame & cs, int & winner );  // winner: 0 / 1, or -1 for a draw
Tile    cs_map_get( const Map & map, int x, int y );         // out of bounds reads as Wall

// Rendering / input (raylib). Call inside BeginDrawing / EndDrawing.
constexpr int CS_TILE_PIXELS = 28;
void            cs_render( const CSGame & cs );
PlayerAction    cs_input_from_keyboard( const CSGame & cs, int playerNumber );   // WASD + mouse aim + LMB
