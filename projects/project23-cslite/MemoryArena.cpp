#include "MemoryArena.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

/*
==========================
arena_create
==========================
*/
Arena arena_create( size_t sizeBytes ) {
    Arena arena = {};
    arena.base = malloc( sizeBytes );
    assert( arena.base != nullptr );
    arena.size = sizeBytes;
    arena.offset = 0;
    return arena;
}

/*
==========================
arena_free
==========================
*/
void arena_free( Arena & arena ) {
    free( arena.base );
    arena = {};
}

/*
==========================
arena_push
==========================
*/
void * arena_push( Arena & arena, size_t sizeBytes, size_t alignment ) {
    assert( arena.base != nullptr );
    assert( alignment != 0 && ( alignment & ( alignment - 1 ) ) == 0 );   // power of two

    uintptr_t start = (uintptr_t) arena.base + arena.offset;
    uintptr_t aligned = ( start + alignment - 1 ) & ~( (uintptr_t) alignment - 1 );
    size_t newOffset = ( aligned - (uintptr_t) arena.base ) + sizeBytes;

    assert( newOffset <= arena.size && "Arena out of memory" );
    if ( newOffset > arena.size ) {
        return nullptr;
    }

    arena.offset = newOffset;
    memset( (void *) aligned, 0, sizeBytes );
    return (void *) aligned;
}

/*
==========================
arena_reset
==========================
*/
void arena_reset( Arena & arena ) {
    arena.offset = 0;
}
