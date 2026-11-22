#pragma once

#define KILOBYTES( x ) (x * 1024);
#define METABYTES( x ) (KILOBYTES( x ) * 1024);
#define GIGABYTES( x ) (METABYTES( x ) * 1024)

/*
===============================================================================
===============================================================================
*/
struct Arena {
    void * base;
    int size;
    int offset;
};

Arena   arena_create( int sizeBytes );
void    arena_free( Arena & arena );
void *  arena_push( Arena & arena, int sizeBytes );
void    arena_reset( Arena & arena );

template<typename T> 
inline T* arena_push(Arena& arena) {
    return (T *) arena_push( sizeof( T ) );
}