#include "sol_pool.h"

namespace sol {

    void PoolSlotsFree( PoolSlots & pool ) {
        ListFree( pool.slots );
        pool.freeHead = 0;
        pool.liveCount = 0;
    }

    void PoolSlotsClear( PoolSlots & pool ) {
        // Rebuilt back to front, so the chain comes out in ascending order and
        // a cleared pool refills from slot 0 rather than in reverse.
        pool.freeHead = 0;
        for ( i32 i = pool.slots.count - 1; i >= 0; i-- ) {
            PoolSlot & slot = pool.slots[i];
            if ( slot.nextFree == kPoolSlotInUse ) {
                // Only a live slot advances: bumping a free one would burn
                // generations every clear for no one's benefit.
                slot.generation++;
                if ( slot.generation == 0 ) {
                    slot.generation = 1;
                }
            }

            slot.nextFree = pool.freeHead;
            pool.freeHead = i + 1;
        }
        pool.liveCount = 0;
    }

    i32 PoolSlotsAcquire( PoolSlots & pool, u32 * outGeneration ) {
        i32 index = -1;

        if ( pool.freeHead != 0 ) {
            index = pool.freeHead - 1;
            PoolSlot & slot = pool.slots[index];
            pool.freeHead = slot.nextFree;
            slot.nextFree = kPoolSlotInUse;
            // Already bumped at release, so handles onto the previous occupant
            // are dead by the time the slot is handed out again.
        } else {
            PoolSlot * slot = ListAddEmpty( pool.slots );
            if ( slot == nullptr ) {
                if ( outGeneration != nullptr ) {
                    *outGeneration = 0;
                }
                return -1;
            }

            // 1 rather than 0: a zeroed Handle has to stay the null handle.
            slot->generation = 1;
            slot->nextFree = kPoolSlotInUse;
            index = pool.slots.count - 1;
        }

        pool.liveCount++;
        if ( outGeneration != nullptr ) {
            *outGeneration = pool.slots[index].generation;
        }
        return index;
    }

    bool PoolSlotsRelease( PoolSlots & pool, i32 index, u32 generation ) {
        if ( !PoolSlotsIsAlive( pool, index, generation ) ) {
            return false;
        }

        PoolSlot & slot = pool.slots[index];
        slot.generation++;
        // Wrapping past 0 would resurrect every handle ever minted against the
        // null handle's generation, so step over it.
        if ( slot.generation == 0 ) {
            slot.generation = 1;
        }

        slot.nextFree = pool.freeHead;
        pool.freeHead = index + 1;
        pool.liveCount--;
        return true;
    }

    bool PoolSlotsIsAlive( const PoolSlots & pool, i32 index, u32 generation ) {
        if ( !ListIsValidIndex( pool.slots, index ) ) {
            return false;
        }

        const PoolSlot & slot = pool.slots[index];
        // The generation test alone would pass for a null handle onto a slot
        // that never issued one, which is why liveness is checked too.
        return slot.nextFree == kPoolSlotInUse && slot.generation == generation;
    }

    bool PoolSlotsIsAliveAt( const PoolSlots & pool, i32 index ) {
        if ( !ListIsValidIndex( pool.slots, index ) ) {
            return false;
        }
        return pool.slots[index].nextFree == kPoolSlotInUse;
    }

} // namespace sol
