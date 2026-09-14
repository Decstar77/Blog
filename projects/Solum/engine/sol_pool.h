#pragma once
#include "sol_defines.h"
#include "sol_list.h"

// Generational handles over a slot array. This exists because a List index is
// not a safe way to name something the renderer owns: the list reallocates, so
// pointers into it dangle, and a removal shifts every index after it, so a
// stored index quietly starts naming a different object. A handle carries the
// generation of the slot it was minted from, so a stale one resolves to null
// instead of to whatever moved in afterwards.

namespace sol {
    template<typename _type_>
    struct Handle {
        u32     index;
        u32     generation;
    };

    template<typename _type_>
    bool HandleIsNull( Handle<_type_> handle ) {
        return handle.generation == 0;
    }

    template<typename _type_>
    bool operator==( Handle<_type_> a, Handle<_type_> b ) {
        return a.index == b.index && a.generation == b.generation;
    }

    template<typename _type_>
    bool operator!=( Handle<_type_> a, Handle<_type_> b ) {
        return !( a == b );
    }

    constexpr i32 kPoolSlotInUse = -1;

    struct PoolSlot {
        u32     generation;
        i32     nextFree;
    };

    struct PoolSlots {
        List<PoolSlot>  slots;
        i32             freeHead;
        i32             liveCount;
    };

    void    PoolSlotsFree( PoolSlots & pool );
    void    PoolSlotsClear( PoolSlots & pool );
    i32     PoolSlotsAcquire( PoolSlots & pool, u32 * outGeneration );
    bool    PoolSlotsRelease( PoolSlots & pool, i32 index, u32 generation );
    bool    PoolSlotsIsAlive( const PoolSlots & pool, i32 index, u32 generation );
    bool    PoolSlotsIsAliveAt( const PoolSlots & pool, i32 index );

    template<typename _type_>
    struct Pool {
        PoolSlots       slots;
        List<_type_>    values;
    };

    template<typename _type_>
    void PoolFree( Pool<_type_> & pool ) {
        PoolSlotsFree( pool.slots );
        ListFree( pool.values );
    }

    template<typename _type_>
    void PoolClear( Pool<_type_> & pool ) {
        PoolSlotsClear( pool.slots );
    }

    template<typename _type_>
    i32 PoolCount( const Pool<_type_> & pool ) {
        return pool.slots.liveCount;
    }

    template<typename _type_>
    i32 PoolSlotCount( const Pool<_type_> & pool ) {
        return pool.slots.slots.count;
    }

    template<typename _type_>
    Handle<_type_> PoolAdd( Pool<_type_> & pool, const _type_ & value ) {
        Handle<_type_> handle = {};

        u32 generation = 0;
        const i32 index = PoolSlotsAcquire( pool.slots, &generation );
        if ( index < 0 ) {
            return handle;
        }

        if ( index >= pool.values.count ) {
            ListResize( pool.values, index + 1 );
            if ( index >= pool.values.count ) {
                PoolSlotsRelease( pool.slots, index, generation );
                return handle;
            }
        }

        pool.values[index] = value;
        handle.index = (u32)index;
        handle.generation = generation;
        return handle;
    }

    template<typename _type_>
    _type_ * PoolGet( Pool<_type_> & pool, Handle<_type_> handle ) {
        if ( !PoolSlotsIsAlive( pool.slots, (i32)handle.index, handle.generation ) ) {
            return nullptr;
        }
        return &pool.values[(i32)handle.index];
    }

    template<typename _type_>
    const _type_ * PoolGet( const Pool<_type_> & pool, Handle<_type_> handle ) {
        if ( !PoolSlotsIsAlive( pool.slots, (i32)handle.index, handle.generation ) ) {
            return nullptr;
        }
        return &pool.values[(i32)handle.index];
    }

    template<typename _type_>
    bool PoolIsValid( const Pool<_type_> & pool, Handle<_type_> handle ) {
        return PoolSlotsIsAlive( pool.slots, (i32)handle.index, handle.generation );
    }

    template<typename _type_>
    bool PoolRemove( Pool<_type_> & pool, Handle<_type_> handle ) {
        return PoolSlotsRelease( pool.slots, (i32)handle.index, handle.generation );
    }

    template<typename _type_>
    _type_ * PoolAt( Pool<_type_> & pool, i32 index ) {
        if ( !PoolSlotsIsAliveAt( pool.slots, index ) ) {
            return nullptr;
        }
        return &pool.values[index];
    }

    template<typename _type_>
    const _type_ * PoolAt( const Pool<_type_> & pool, i32 index ) {
        if ( !PoolSlotsIsAliveAt( pool.slots, index ) ) {
            return nullptr;
        }
        return &pool.values[index];
    }

    template<typename _type_>
    Handle<_type_> PoolHandleAt( const Pool<_type_> & pool, i32 index ) {
        Handle<_type_> handle = {};
        if ( !PoolSlotsIsAliveAt( pool.slots, index ) ) {
            return handle;
        }

        handle.index = (u32)index;
        handle.generation = pool.slots.slots[index].generation;
        return handle;
    }

} // namespace sol
