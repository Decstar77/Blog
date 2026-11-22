#include "CSGame.h"

#include "raylib.h"

#include <cmath>

/*
===============================================================================
Simulation
===============================================================================
*/

/*
==========================
cs_map_get
==========================
*/
Tile cs_map_get( const Map & map, int x, int y ) {
    if ( x < 0 || y < 0 || x >= map.width || y >= map.height ) {
        return Tile::Wall;
    }
    return map.tiles[ y * map.width + x ];
}

/*
==========================
map_create_empty
Empty floor with a one-tile wall around the border.
==========================
*/
static Map map_create_empty( Arena & arena, int width, int height ) {
    Map map = {};
    map.width = width;
    map.height = height;
    map.tiles = arena_push_array<Tile>( arena, (size_t) width * height );

    for ( int y = 0; y < height; y++ ) {
        for ( int x = 0; x < width; x++ ) {
            bool border = x == 0 || y == 0 || x == width - 1 || y == height - 1;
            map.tiles[ y * width + x ] = border ? Tile::Wall : Tile::Empty;
        }
    }
    return map;
}

/*
==========================
cs_new_game
==========================
*/
void cs_new_game( CSGame & cs ) {
    if ( cs.arena.base == nullptr ) {
        cs.arena = arena_create( MEGABYTES( 1 ) );
    } else {
        arena_reset( cs.arena );
    }

    cs.map = map_create_empty( cs.arena, CS_MAP_WIDTH, CS_MAP_HEIGHT );
    cs.bullets = arena_push_array<Bullet>( cs.arena, CS_MAX_BULLETS );
    cs.bulletCount = 0;
    cs.tick = 0;

    // Spawn facing each other on opposite sides of the map
    float midY = CS_MAP_HEIGHT * 0.5f;
    cs.players[ 0 ] = { 0, CS_PLAYER_HEALTH, { 3.0f, midY }, 0.0f, 0.0f };
    cs.players[ 1 ] = { 1, CS_PLAYER_HEALTH, { CS_MAP_WIDTH - 3.0f, midY }, 3.14159265f, 0.0f };
}

/*
==========================
cs_free
==========================
*/
void cs_free( CSGame & cs ) {
    arena_free( cs.arena );
    cs = {};
}

/*
==========================
collide_circle_with_walls
Pushes a circle out of any wall tile it overlaps.
==========================
*/
static void collide_circle_with_walls( const Map & map, Vec2 & pos, float radius ) {
    int minX = (int) floorf( pos.x - radius );
    int maxX = (int) floorf( pos.x + radius );
    int minY = (int) floorf( pos.y - radius );
    int maxY = (int) floorf( pos.y + radius );

    for ( int ty = minY; ty <= maxY; ty++ ) {
        for ( int tx = minX; tx <= maxX; tx++ ) {
            if ( cs_map_get( map, tx, ty ) != Tile::Wall ) {
                continue;
            }

            // Closest point on the tile's box to the circle centre
            float cx = fminf( fmaxf( pos.x, (float) tx ), (float) tx + 1.0f );
            float cy = fminf( fmaxf( pos.y, (float) ty ), (float) ty + 1.0f );
            float dx = pos.x - cx;
            float dy = pos.y - cy;
            float distSq = dx * dx + dy * dy;

            if ( distSq >= radius * radius || distSq == 0.0f ) {
                // distSq == 0 means the centre is inside the tile; can't happen
                // at our speeds (one step moves less than the radius).
                continue;
            }

            float dist = sqrtf( distSq );
            float push = radius - dist;
            pos.x += dx / dist * push;
            pos.y += dy / dist * push;
        }
    }
}

/*
==========================
player_apply_action
==========================
*/
static void player_apply_action( CSGame & cs, Player & player, const PlayerAction & action ) {
    if ( player.health <= 0 ) {
        return;
    }

    // Movement
    float mx = (float) action.moveRight - (float) action.moveLeft;
    float my = (float) action.moveDown - (float) action.moveUp;
    float len = sqrtf( mx * mx + my * my );
    if ( len > 0.0f ) {
        player.pos.x += mx / len * CS_PLAYER_SPEED * CS_DT;
        player.pos.y += my / len * CS_PLAYER_SPEED * CS_DT;
        collide_circle_with_walls( cs.map, player.pos, CS_PLAYER_RADIUS );
    }

    // Aim
    player.rotation = action.rotation;

    // Shoot
    player.shootCooldown = fmaxf( 0.0f, player.shootCooldown - CS_DT );
    if ( action.shoot && player.shootCooldown <= 0.0f && cs.bulletCount < CS_MAX_BULLETS ) {
        Vec2 dir = { cosf( player.rotation ), sinf( player.rotation ) };
        float muzzle = CS_PLAYER_RADIUS + CS_BULLET_RADIUS + 0.05f;

        Bullet & b = cs.bullets[ cs.bulletCount++ ];
        b.pos = { player.pos.x + dir.x * muzzle, player.pos.y + dir.y * muzzle };
        b.vel = { dir.x * CS_BULLET_SPEED, dir.y * CS_BULLET_SPEED };
        b.owner = player.playerNumber;

        player.shootCooldown = CS_SHOOT_COOLDOWN;
    }
}

/*
==========================
update_bullets
==========================
*/
static void update_bullets( CSGame & cs ) {
    int i = 0;
    while ( i < cs.bulletCount ) {
        Bullet & b = cs.bullets[ i ];
        b.pos.x += b.vel.x * CS_DT;
        b.pos.y += b.vel.y * CS_DT;

        bool dead = cs_map_get( cs.map, (int) floorf( b.pos.x ), (int) floorf( b.pos.y ) ) == Tile::Wall;

        for ( Player & p : cs.players ) {
            if ( dead || p.playerNumber == b.owner || p.health <= 0 ) {
                continue;
            }
            float dx = p.pos.x - b.pos.x;
            float dy = p.pos.y - b.pos.y;
            float r = CS_PLAYER_RADIUS + CS_BULLET_RADIUS;
            if ( dx * dx + dy * dy < r * r ) {
                p.health -= CS_BULLET_DAMAGE;
                dead = true;
            }
        }

        if ( dead ) {
            cs.bullets[ i ] = cs.bullets[ --cs.bulletCount ];   // swap-remove, re-check slot i
        } else {
            i++;
        }
    }
}

/*
==========================
cs_step
==========================
*/
void cs_step( CSGame & cs, PlayerAction p1, PlayerAction p2 ) {
    int winner;
    if ( cs_is_terminal( cs, winner ) ) {
        return;
    }

    player_apply_action( cs, cs.players[ 0 ], p1 );
    player_apply_action( cs, cs.players[ 1 ], p2 );
    update_bullets( cs );
    cs.tick++;
}

/*
==========================
cs_is_terminal
==========================
*/
bool cs_is_terminal( const CSGame & cs, int & winner ) {
    bool dead0 = cs.players[ 0 ].health <= 0;
    bool dead1 = cs.players[ 1 ].health <= 0;

    winner = -1;
    if ( dead0 && dead1 ) {
        return true;
    }
    if ( dead0 ) {
        winner = 1;
        return true;
    }
    if ( dead1 ) {
        winner = 0;
        return true;
    }
    return cs.tick >= CS_MAX_TICKS;
}

/*
===============================================================================
Rendering / input (raylib)
===============================================================================
*/

static Vector2 to_screen( Vec2 p ) {
    return { p.x * CS_TILE_PIXELS, p.y * CS_TILE_PIXELS };
}

/*
==========================
cs_render
==========================
*/
void cs_render( const CSGame & cs ) {
    const Color floorColor = { 40, 44, 52, 255 };
    const Color gridColor = { 50, 55, 64, 255 };
    const Color wallColor = { 120, 124, 135, 255 };
    const Color playerColors[ 2 ] = { { 80, 160, 255, 255 }, { 255, 90, 80, 255 } };

    ClearBackground( floorColor );

    // Map
    for ( int y = 0; y < cs.map.height; y++ ) {
        for ( int x = 0; x < cs.map.width; x++ ) {
            int px = x * CS_TILE_PIXELS;
            int py = y * CS_TILE_PIXELS;
            if ( cs_map_get( cs.map, x, y ) == Tile::Wall ) {
                DrawRectangle( px, py, CS_TILE_PIXELS, CS_TILE_PIXELS, wallColor );
            } else {
                DrawRectangleLines( px, py, CS_TILE_PIXELS, CS_TILE_PIXELS, gridColor );
            }
        }
    }

    // Bullets
    for ( int i = 0; i < cs.bulletCount; i++ ) {
        DrawCircleV( to_screen( cs.bullets[ i ].pos ), CS_BULLET_RADIUS * CS_TILE_PIXELS, YELLOW );
    }

    // Players
    for ( const Player & p : cs.players ) {
        Vector2 centre = to_screen( p.pos );
        float r = CS_PLAYER_RADIUS * CS_TILE_PIXELS;
        Color c = p.health > 0 ? playerColors[ p.playerNumber ] : GRAY;

        Vector2 barrelEnd = { centre.x + cosf( p.rotation ) * r * 1.6f, centre.y + sinf( p.rotation ) * r * 1.6f };
        DrawLineEx( centre, barrelEnd, 4.0f, c );
        DrawCircleV( centre, r, c );

        // Health bar
        float barW = r * 2.4f;
        float frac = fmaxf( 0.0f, (float) p.health / CS_PLAYER_HEALTH );
        Rectangle bg = { centre.x - barW * 0.5f, centre.y - r - 10.0f, barW, 4.0f };
        DrawRectangleRec( bg, DARKGRAY );
        DrawRectangleRec( { bg.x, bg.y, bg.width * frac, bg.height }, GREEN );
    }

    // HUD
    float timeLeft = ( CS_MAX_TICKS - cs.tick ) * CS_DT;
    DrawText( TextFormat( "%.1fs", timeLeft ), 6, 6, 16, RAYWHITE );

    int winner;
    if ( cs_is_terminal( cs, winner ) ) {
        const char * msg = winner < 0 ? "Draw" : ( winner == 0 ? "Blue wins" : "Red wins" );
        int screenW = cs.map.width * CS_TILE_PIXELS;
        int screenH = cs.map.height * CS_TILE_PIXELS;
        int w = MeasureText( msg, 40 );
        DrawText( msg, ( screenW - w ) / 2, screenH / 2 - 40, 40, RAYWHITE );
        const char * hint = "Press R to restart";
        w = MeasureText( hint, 20 );
        DrawText( hint, ( screenW - w ) / 2, screenH / 2 + 10, 20, LIGHTGRAY );
    }
}

/*
==========================
cs_input_from_keyboard
==========================
*/
PlayerAction cs_input_from_keyboard( const CSGame & cs, int playerNumber ) {
    const Player & p = cs.players[ playerNumber ];

    PlayerAction a = {};
    a.moveLeft = IsKeyDown( KEY_A );
    a.moveRight = IsKeyDown( KEY_D );
    a.moveUp = IsKeyDown( KEY_W );
    a.moveDown = IsKeyDown( KEY_S );
    a.shoot = IsMouseButtonDown( MOUSE_BUTTON_LEFT );

    Vector2 mouse = GetMousePosition();
    Vector2 centre = to_screen( p.pos );
    a.rotation = atan2f( mouse.y - centre.y, mouse.x - centre.x );
    return a;
}
