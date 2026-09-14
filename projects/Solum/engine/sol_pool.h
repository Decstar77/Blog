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

    // Templated purely so a RenderStaticMesh handle cannot be passed where a
    // RenderTexture handle is wanted. _type_ is never dereferenced here, so it
    // may be incomplete at the point the handle type is named.
    template<typename _type_>
    struct Handle {
        u32     index;
        u32     generation;
    };

    // Slot generations start at 1, so a zeroed Handle is the null handle and
    // there is no sentinel value for callers to remember. {} means nothing.
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

    // A live slot's nextFree, which is never a valid link.
    constexpr i32 kPoolSlotInUse = -1;

    struct PoolSlot {
        // Bumped on every release, so handles minted before it stop resolving.
        // Never 0 on a live slot - that value is reserved for the null handle.
        u32     generation;
        // One-based link into the free list, 0 ending the chain, or
        // kPoolSlotInUse while the slot holds something.
        i32     nextFree;
    };

    // The type-agnostic half of a Pool: which slots are live, what generation
    // each is on, and which one the next add lands in. Split out so the
    // bookkeeping is compiled once rather than once per pooled type.
    struct PoolSlots {
        List<PoolSlot>  slots;
        // One-based, so 0 means an empty free list rather than slot 0. That is
        // what lets a zeroed PoolSlots be a valid empty pool: a plain index
        // here would have {} claiming slot 0 was free.
        i32             freeHead;
        i32             liveCount;
    };

    void    PoolSlotsFree( PoolSlots & pool );
    // Kills every live slot and hands them all back. Keeps the allocation, and
    // bumps generations, so handles from before this do not resolve after it.
    void    PoolSlotsClear( PoolSlots & pool );
    // The slot index, or -1 if the array could not grow. outGeneration is the
    // generation the caller should mint its handle with.
    i32     PoolSlotsAcquire( PoolSlots & pool, u32 * outGeneration );
    // False when the handle was already stale, which makes a double release a
    // no-op rather than a second bump that would revive an older handle.
    bool    PoolSlotsRelease( PoolSlots & pool, i32 index, u32 generation );
    bool    PoolSlotsIsAlive( const PoolSlots & pool, i32 index, u32 generation );
    bool    PoolSlotsIsAliveAt( const PoolSlots & pool, i32 index );

    // Storage plus the bookkeeping above. values runs parallel to slots, so a
    // slot index is a values index; dead entries hold whatever the last
    // occupant left behind and must not be read without an aliveness check.
    // A zeroed Pool is a valid empty pool.
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

    // Keeps both allocations, drops every element.
    template<typename _type_>
    void PoolClear( Pool<_type_> & pool ) {
        PoolSlotsClear( pool.slots );
    }

    // Live elements, not slots.
    template<typename _type_>
    i32 PoolCount( const Pool<_type_> & pool ) {
        return pool.slots.liveCount;
    }

    // Slots ever handed out, live or not. The bound for an index walk.
    template<typename _type_>
    i32 PoolSlotCount( const Pool<_type_> & pool ) {
        return pool.slots.slots.count;
    }

    // Null handle if the pool could not grow, so the result is worth checking
    // on the paths where an allocation failure matters.
    template<typename _type_>
    Handle<_type_> PoolAdd( Pool<_type_> & pool, const _type_ & value ) {
        Handle<_type_> handle = {};

        u32 generation = 0;
        const i32 index = PoolSlotsAcquire( pool.slots, &generation );
        if ( index < 0 ) {
            return handle;
        }

        // Only a brand new slot outruns values; a reused one already has a
        // parallel element sitting there to overwrite.
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

    // Null for a stale or null handle, which is the whole point of the pool:
    // every read goes through a check the caller cannot skip by accident.
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

    // False when the handle was already stale. The element itself is left
    // alone: anything holding resources must be torn down before this.
    template<typename _type_>
    bool PoolRemove( Pool<_type_> & pool, Handle<_type_> handle ) {
        return PoolSlotsRelease( pool.slots, (i32)handle.index, handle.generation );
    }

    // Walking a pool is a loop to PoolSlotCount that skips the nulls these
    // return for dead slots.
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

    // The handle naming a slot found by walking, so an iteration can remove
    // what it finds. Null for a dead slot.
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
