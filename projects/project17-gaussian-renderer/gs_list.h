#pragma once

#include "gs_defines.h"

#include <cstdlib>
#include <cstring>

template <typename _type_>
struct List {
    _type_ * data;
    i32 count;
    i32 cap;

    _type_ & operator[]( i32 index ) { return data[index]; }
    const _type_ & operator[]( i32 index ) const { return data[index]; }

    _type_ * begin() { return data; }
    _type_ * end() { return data + count; }
    const _type_ * begin() const { return data; }
    const _type_ * end() const { return data + count; }
};

constexpr i32 kListFirstCap = 8;

/*
===================
===================
*/
template <typename _type_>
List<_type_> list_create() {
    List<_type_> list = {};
    return list;
}

/*
===================
===================
*/
template <typename _type_>
void list_free( List<_type_> & list ) {
    ::free( list.data );
    list.data = nullptr;
    list.count = 0;
    list.cap = 0;
}

/*
===================
===================
*/
template <typename _type_>
void list_clear( List<_type_> & list ) {
    list.count = 0;
}

/*
===================
===================
*/
template <typename _type_>
bool list_is_empty( const List<_type_> & list ) {
    return list.count == 0;
}

/*
===================
===================
*/
template <typename _type_>
bool list_is_full( const List<_type_> & list ) {
    return list.count == list.cap;
}

/*
===================
===================
*/
template <typename _type_>
bool list_is_valid_index( const List<_type_> & list, i32 index ) {
    return index >= 0 && index < list.count;
}

/*
===================
===================
*/
template <typename _type_>
void list_reserve( List<_type_> & list, i32 wanted_cap ) {
    if ( wanted_cap <= list.cap ) {
        return;
    }

    i32 new_cap = list.cap != 0 ? list.cap : kListFirstCap;
    while ( new_cap < wanted_cap ) {
        new_cap *= 2;
    }

    _type_ * new_data = (_type_ *) ::realloc( list.data, (u64) new_cap * sizeof( _type_ ) );
    if ( new_data == nullptr ) {
        return;
    }

    list.data = new_data;
    list.cap = new_cap;
}

/*
===================
===================
*/
template <typename _type_>
void list_shrink_to_fit( List<_type_> & list ) {
    if ( list.cap == list.count ) {
        return;
    }

    if ( list.count == 0 ) {
        list_free( list );
        return;
    }

    _type_ * new_data = (_type_ *) ::realloc( list.data, (u64) list.count * sizeof( _type_ ) );
    if ( new_data == nullptr ) {
        return;
    }

    list.data = new_data;
    list.cap = list.count;
}

/*
===================
===================
*/
template <typename _type_>
_type_ * list_add( List<_type_> & list, const _type_ & value ) {
    list_reserve( list, list.count + 1 );
    if ( list.count == list.cap ) {
        return nullptr;
    }

    list.data[list.count] = value;
    list.count++;
    return &list.data[list.count - 1];
}

/*
===================
===================
*/
template <typename _type_>
_type_ * list_add_empty( List<_type_> & list ) {
    list_reserve( list, list.count + 1 );
    if ( list.count == list.cap ) {
        return nullptr;
    }

    list.count++;
    return &list.data[list.count - 1];
}

/*
===================
===================
*/
template <typename _type_>
void list_add_range( List<_type_> & list, const _type_ * values, i32 value_count ) {
    if ( value_count <= 0 ) {
        return;
    }

    list_reserve( list, list.count + value_count );
    if ( list.cap - list.count < value_count ) {
        return;
    }

    ::memcpy( list.data + list.count, values, (u64) value_count * sizeof( _type_ ) );
    list.count += value_count;
}

/*
===================
===================
*/
template <typename _type_>
void list_add_range( List<_type_> & list, const List<_type_> & other ) {
    list_add_range( list, other.data, other.count );
}

/*
===================
===================
*/
template <typename _type_>
_type_ * list_insert( List<_type_> & list, i32 index, const _type_ & value ) {
    if ( index < 0 || index > list.count ) {
        return nullptr;
    }

    list_reserve( list, list.count + 1 );
    if ( list.count == list.cap ) {
        return nullptr;
    }

    i32 tail = list.count - index;
    if ( tail > 0 ) {
        ::memmove( list.data + index + 1, list.data + index, (u64) tail * sizeof( _type_ ) );
    }

    list.data[index] = value;
    list.count++;
    return &list.data[index];
}

/*
===================
===================
*/
template <typename _type_>
void list_remove_index( List<_type_> & list, i32 index ) {
    if ( list_is_valid_index( list, index ) == false ) {
        return;
    }

    i32 tail = list.count - index - 1;
    if ( tail > 0 ) {
        ::memmove( list.data + index, list.data + index + 1, (u64) tail * sizeof( _type_ ) );
    }

    list.count--;
}

/*
===================
===================
*/
template <typename _type_>
void list_remove_index_fast( List<_type_> & list, i32 index ) {
    if ( list_is_valid_index( list, index ) == false ) {
        return;
    }

    list.data[index] = list.data[list.count - 1];
    list.count--;
}

/*
===================
===================
*/
template <typename _type_>
void list_remove_range( List<_type_> & list, i32 index, i32 remove_count ) {
    if ( list_is_valid_index( list, index ) == false || remove_count <= 0 ) {
        return;
    }

    if ( remove_count > list.count - index ) {
        remove_count = list.count - index;
    }

    i32 tail = list.count - index - remove_count;
    if ( tail > 0 ) {
        ::memmove( list.data + index, list.data + index + remove_count,
            (u64) tail * sizeof( _type_ ) );
    }

    list.count -= remove_count;
}

/*
===================
===================
*/
template <typename _type_>
_type_ list_pop( List<_type_> & list ) {
    if ( list.count == 0 ) {
        _type_ empty = {};
        return empty;
    }

    list.count--;
    return list.data[list.count];
}

/*
===================
===================
*/
template <typename _type_>
_type_ * list_get( List<_type_> & list, i32 index ) {
    if ( list_is_valid_index( list, index ) == false ) {
        return nullptr;
    }
    return &list.data[index];
}

/*
===================
===================
*/
template <typename _type_>
_type_ * list_last( List<_type_> & list ) {
    if ( list.count == 0 ) {
        return nullptr;
    }
    return &list.data[list.count - 1];
}

/*
===================
===================
*/
template <typename _type_>
i32 list_index_of( const List<_type_> & list, const _type_ & value ) {
    for ( i32 i = 0; i < list.count; i++ ) {
        if ( list.data[i] == value ) {
            return i;
        }
    }
    return -1;
}

/*
===================
===================
*/
template <typename _type_>
bool list_contains( const List<_type_> & list, const _type_ & value ) {
    return list_index_of( list, value ) >= 0;
}

/*
===================
===================
*/
template <typename _type_>
bool list_remove_value( List<_type_> & list, const _type_ & value ) {
    i32 index = list_index_of( list, value );
    if ( index < 0 ) {
        return false;
    }

    list_remove_index( list, index );
    return true;
}

/*
===================
===================
*/
template <typename _type_>
bool list_remove_value_fast( List<_type_> & list, const _type_ & value ) {
    i32 index = list_index_of( list, value );
    if ( index < 0 ) {
        return false;
    }

    list_remove_index_fast( list, index );
    return true;
}

/*
===================
===================
*/
template <typename _type_, typename _pred_>
i32 list_index_of_pred( const List<_type_> & list, _pred_ pred ) {
    for ( i32 i = 0; i < list.count; i++ ) {
        if ( pred( list.data[i] ) ) {
            return i;
        }
    }
    return -1;
}

/*
===================
===================
*/
template <typename _type_, typename _pred_>
_type_ * list_find( List<_type_> & list, _pred_ pred ) {
    i32 index = list_index_of_pred( list, pred );
    if ( index < 0 ) {
        return nullptr;
    }
    return &list.data[index];
}

/*
===================
===================
*/
template <typename _type_, typename _pred_>
i32 list_remove_if( List<_type_> & list, _pred_ pred ) {
    i32 write = 0;
    for ( i32 read = 0; read < list.count; read++ ) {
        if ( pred( list.data[read] ) ) {
            continue;
        }

        if ( write != read ) {
            list.data[write] = list.data[read];
        }
        write++;
    }

    i32 removed = list.count - write;
    list.count = write;
    return removed;
}

/*
===================
===================
*/
template <typename _type_>
void list_resize( List<_type_> & list, i32 new_count ) {
    if ( new_count < 0 ) {
        new_count = 0;
    }

    if ( new_count <= list.count ) {
        list.count = new_count;
        return;
    }

    list_reserve( list, new_count );
    if ( list.cap < new_count ) {
        return;
    }

    ::memset( list.data + list.count, 0, (u64) ( new_count - list.count ) * sizeof( _type_ ) );
    list.count = new_count;
}

/*
===================
===================
*/
template <typename _type_>
void list_fill( List<_type_> & list, const _type_ & value ) {
    for ( i32 i = 0; i < list.count; i++ ) {
        list.data[i] = value;
    }
}

/*
===================
===================
*/
template <typename _type_>
List<_type_> list_copy( const List<_type_> & list ) {
    List<_type_> result = {};
    list_add_range( result, list.data, list.count );
    return result;
}

/*
===================
===================
*/
template <typename _type_>
void list_swap( List<_type_> & list, i32 a, i32 b ) {
    if ( list_is_valid_index( list, a ) == false || list_is_valid_index( list, b ) == false ) {
        return;
    }

    _type_ temp = list.data[a];
    list.data[a] = list.data[b];
    list.data[b] = temp;
}

/*
===================
===================
*/
template <typename _type_>
void list_reverse( List<_type_> & list ) {
    for ( i32 i = 0, j = list.count - 1; i < j; i++, j-- ) {
        list_swap( list, i, j );
    }
}
