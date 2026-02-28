use crate::arena::kv_arena::KvArena;
use crate::arena::position_map::PositionMap;

/// Compact the arena: move all occupied slots to a contiguous prefix.
/// Delegates to `KvArena::compact()`.
pub fn compact(arena: &mut KvArena) -> PositionMap {
    arena.compact()
}
