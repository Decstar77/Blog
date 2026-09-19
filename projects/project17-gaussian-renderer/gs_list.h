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

template <typename _type_>
List<_type_> ListCreate() {
    List<_type_> list = {};
    return list;
}

template <typename _type_>
void ListFree( List<_type_> & list ) {
    ::free( list.data );
    list.data = nullptr;
    list.count = 0;
    list.cap = 0;
}

template <typename _type_>
void ListClear( List<_type_> & list ) {
    list.count = 0;
}

template <typename _type_>
bool ListIsEmpty( const List<_type_> & list ) {
    return list.count == 0;
}

template <typename _type_>
bool ListIsFull( const List<_type_> & list ) {
    return list.count == list.cap;
}

template <typename _type_>
bool ListIsValidIndex( const List<_type_> & list, i32 index ) {
    return index >= 0 && index < list.count;
}

template <typename _type_>
void ListReserve( List<_type_> & list, i32 wantedCap ) {
    if ( wantedCap <= list.cap ) {
        return;
    }

    i32 newCap = list.cap != 0 ? list.cap : kListFirstCap;
    while ( newCap < wantedCap ) {
        newCap *= 2;
    }

    _type_ * newData = (_type_ *) ::realloc( list.data, (u64) newCap * sizeof( _type_ ) );
    if ( newData == nullptr ) {
        return;
    }

    list.data = newData;
    list.cap = newCap;
}

template <typename _type_>
void ListShrinkToFit( List<_type_> & list ) {
    if ( list.cap == list.count ) {
        return;
    }

    if ( list.count == 0 ) {
        ListFree( list );
        return;
    }

    _type_ * newData = (_type_ *) ::realloc( list.data, (u64) list.count * sizeof( _type_ ) );
    if ( newData == nullptr ) {
        return;
    }

    list.data = newData;
    list.cap = list.count;
}

template <typename _type_>
_type_ * ListAdd( List<_type_> & list, const _type_ & value ) {
    ListReserve( list, list.count + 1 );
    if ( list.count == list.cap ) {
        return nullptr;
    }

    list.data[list.count] = value;
    list.count++;
    return &list.data[list.count - 1];
}

template <typename _type_>
_type_ * ListAddEmpty( List<_type_> & list ) {
    ListReserve( list, list.count + 1 );
    if ( list.count == list.cap ) {
        return nullptr;
    }

    list.count++;
    return &list.data[list.count - 1];
}

template <typename _type_>
void ListAddRange( List<_type_> & list, const _type_ * values, i32 valueCount ) {
    if ( valueCount <= 0 ) {
        return;
    }

    ListReserve( list, list.count + valueCount );
    if ( list.cap - list.count < valueCount ) {
        return;
    }

    ::memcpy( list.data + list.count, values, (u64) valueCount * sizeof( _type_ ) );
    list.count += valueCount;
}

template <typename _type_>
void ListAddRange( List<_type_> & list, const List<_type_> & other ) {
    ListAddRange( list, other.data, other.count );
}

template <typename _type_>
_type_ * ListInsert( List<_type_> & list, i32 index, const _type_ & value ) {
    if ( index < 0 || index > list.count ) {
        return nullptr;
    }

    ListReserve( list, list.count + 1 );
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

template <typename _type_>
void ListRemoveIndex( List<_type_> & list, i32 index ) {
    if ( ListIsValidIndex( list, index ) == false ) {
        return;
    }

    i32 tail = list.count - index - 1;
    if ( tail > 0 ) {
        ::memmove( list.data + index, list.data + index + 1, (u64) tail * sizeof( _type_ ) );
    }

    list.count--;
}

template <typename _type_>
void ListRemoveIndexFast( List<_type_> & list, i32 index ) {
    if ( ListIsValidIndex( list, index ) == false ) {
        return;
    }

    list.data[index] = list.data[list.count - 1];
    list.count--;
}

template <typename _type_>
void ListRemoveRange( List<_type_> & list, i32 index, i32 removeCount ) {
    if ( ListIsValidIndex( list, index ) == false || removeCount <= 0 ) {
        return;
    }

    if ( removeCount > list.count - index ) {
        removeCount = list.count - index;
    }

    i32 tail = list.count - index - removeCount;
    if ( tail > 0 ) {
        ::memmove( list.data + index, list.data + index + removeCount,
            (u64) tail * sizeof( _type_ ) );
    }

    list.count -= removeCount;
}

template <typename _type_>
_type_ ListPop( List<_type_> & list ) {
    if ( list.count == 0 ) {
        _type_ empty = {};
        return empty;
    }

    list.count--;
    return list.data[list.count];
}

template <typename _type_>
_type_ * ListGet( List<_type_> & list, i32 index ) {
    if ( ListIsValidIndex( list, index ) == false ) {
        return nullptr;
    }
    return &list.data[index];
}

template <typename _type_>
_type_ * ListLast( List<_type_> & list ) {
    if ( list.count == 0 ) {
        return nullptr;
    }
    return &list.data[list.count - 1];
}

template <typename _type_>
i32 ListIndexOf( const List<_type_> & list, const _type_ & value ) {
    for ( i32 i = 0; i < list.count; i++ ) {
        if ( list.data[i] == value ) {
            return i;
        }
    }
    return -1;
}

template <typename _type_>
bool ListContains( const List<_type_> & list, const _type_ & value ) {
    return ListIndexOf( list, value ) >= 0;
}

template <typename _type_>
bool ListRemoveValue( List<_type_> & list, const _type_ & value ) {
    i32 index = ListIndexOf( list, value );
    if ( index < 0 ) {
        return false;
    }

    ListRemoveIndex( list, index );
    return true;
}

template <typename _type_>
bool ListRemoveValueFast( List<_type_> & list, const _type_ & value ) {
    i32 index = ListIndexOf( list, value );
    if ( index < 0 ) {
        return false;
    }

    ListRemoveIndexFast( list, index );
    return true;
}

template <typename _type_, typename _pred_>
i32 ListIndexOfPred( const List<_type_> & list, _pred_ pred ) {
    for ( i32 i = 0; i < list.count; i++ ) {
        if ( pred( list.data[i] ) ) {
            return i;
        }
    }
    return -1;
}

template <typename _type_, typename _pred_>
_type_ * ListFind( List<_type_> & list, _pred_ pred ) {
    i32 index = ListIndexOfPred( list, pred );
    if ( index < 0 ) {
        return nullptr;
    }
    return &list.data[index];
}

template <typename _type_, typename _pred_>
i32 ListRemoveIf( List<_type_> & list, _pred_ pred ) {
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

template <typename _type_>
void ListResize( List<_type_> & list, i32 newCount ) {
    if ( newCount < 0 ) {
        newCount = 0;
    }

    if ( newCount <= list.count ) {
        list.count = newCount;
        return;
    }

    ListReserve( list, newCount );
    if ( list.cap < newCount ) {
        return;
    }

    ::memset( list.data + list.count, 0, (u64) ( newCount - list.count ) * sizeof( _type_ ) );
    list.count = newCount;
}

template <typename _type_>
void ListFill( List<_type_> & list, const _type_ & value ) {
    for ( i32 i = 0; i < list.count; i++ ) {
        list.data[i] = value;
    }
}

template <typename _type_>
List<_type_> ListCopy( const List<_type_> & list ) {
    List<_type_> result = {};
    ListAddRange( result, list.data, list.count );
    return result;
}

template <typename _type_>
void ListSwap( List<_type_> & list, i32 a, i32 b ) {
    if ( ListIsValidIndex( list, a ) == false || ListIsValidIndex( list, b ) == false ) {
        return;
    }

    _type_ temp = list.data[a];
    list.data[a] = list.data[b];
    list.data[b] = temp;
}

template <typename _type_>
void ListReverse( List<_type_> & list ) {
    for ( i32 i = 0, j = list.count - 1; i < j; i++, j-- ) {
        ListSwap( list, i, j );
    }
}
