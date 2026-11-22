#pragma once

#include <cstddef>

#define KILOBYTES( x ) ( (size_t) ( x ) * 1024 )
#define MEGABYTES( x ) ( KILOBYTES( x ) * 1024 )
#define GIGABYTES( x ) ( MEGABYTES( x ) * 1024 )

/*
===============================================================================
Linear (bump) allocator. One big block up front, pushes just move an offset
forward. Nothing is freed individually: reset the whole arena or free it.
===============================================================================
*/
struct Arena {
    void * base = nullptr;
    size_t size = 0;
    size_t offset = 0;
};

Arena   arena_create( size_t sizeBytes );
void    arena_free( Arena & arena );
void *  arena_push( Arena & arena, size_t sizeBytes, size_t alignment = 16 );  // zeroed memory, asserts on overflow
void    arena_reset( Arena & arena );

template<typename T>
inline T * arena_push( Arena & arena ) {
    return (T *) arena_push( arena, sizeof( T ), alignof( T ) );
}

template<typename T>
inline T * arena_push_array( Arena & arena, size_t count ) {
    return (T *) arena_push( arena, sizeof( T ) * count, alignof( T ) );
}
