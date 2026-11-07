//! CPU-side bookkeeping for a sparse, effectively unbounded Life world.
//!
//! The plane is cut into `TILE` x `TILE` tiles. Only tiles near live cells exist;
//! each one occupies a slot in a fixed-size GPU pool. This module decides which
//! tiles exist and where they live; the GPU does the actual simulation.
//!
//! # Why tiles never need to exist "just in time"
//!
//! Life spreads at most one cell per generation. If a tile has a live cell within
//! `MARGIN` cells of an edge, we make sure the neighbour across that edge exists.
//! The compute shader reports, per tile, whether it is alive and which neighbours
//! it needs (see [`flags_for`]), and the CPU looks at those flags every
//! `BOOKKEEPING_INTERVAL` generations.
//!
//! Readbacks are pipelined: flags taken at one bookkeeping point are applied at the
//! next, while the GPU keeps running. So a tile allocation may be based on flags
//! up to `2 * BOOKKEEPING_INTERVAL` generations old, and `MARGIN` must be at least
//! that: a live cell farther than `MARGIN` from an edge can't influence the tile
//! beyond it for more than `MARGIN` generations.

use std::collections::{HashMap, HashSet};

/// Tile edge length in cells.
pub const TILE: u32 = 128;
pub const TILE_CELLS: usize = (TILE * TILE) as usize;
/// Cells are bit-packed: a tile is TILE rows of TILE / 32 `u32` words. Cell index
/// `y * TILE + x` (see [`locate`]) lives in word `index / 32`, bit `index % 32`.
pub const TILE_WORDS: usize = TILE_CELLS / 32;
/// Distance from a tile edge that counts as "near" it. Must be <= TILE / 2, and
/// must match MARGIN in sparse_life.wgsl. (TILE must match both shaders too.)
pub const MARGIN: u32 = 64;
/// Generations between bookkeeping passes. Readbacks are applied one pass late, so
/// this must be at most MARGIN / 2.
pub const BOOKKEEPING_INTERVAL: u32 = MARGIN / 2;

// The shaders rely on these: rows are whole words (TILE % 32), the render shader
// divides by shifting (power of two), and a word is never near both the west and
// east edges (MARGIN % 32). Pipelined readbacks need 2 * interval <= MARGIN.
const _: () = assert!(TILE.is_power_of_two() && TILE % 32 == 0);
const _: () = assert!(MARGIN % 32 == 0 && MARGIN <= TILE / 2);
const _: () = assert!(2 * BOOKKEEPING_INTERVAL <= MARGIN);

/// WGSL `const` declarations for the tile constants, prepended to the sparse
/// shaders so this file is the only place they're defined.
pub fn shader_constants() -> String {
    format!(
        "const TILE: u32 = {TILE}u;\n\
         const TILE_SHIFT: u32 = {}u;\n\
         const MARGIN: u32 = {MARGIN}u;\n\
         const POP_SHIFT: u32 = {POP_SHIFT}u;\n",
        TILE.trailing_zeros()
    )
}
/// Neighbour-table entry for "no tile here".
pub const EMPTY: u32 = u32::MAX;
/// Flag bit: the tile has at least one live cell.
pub const ALIVE: u32 = 1;
/// Flags bits 16.. hold the tile's live-cell count.
pub const POP_SHIFT: u32 = 16;
/// Neighbour directions as (dx, dy), y pointing down. Flag bit `1 << (1 + d)` means
/// "tile needs neighbour `d`" (see [`need_bit`]); [`opposite`] flips a direction.
pub const DIRS: [(i32, i32); 8] =
    [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

pub type TileCoord = (i32, i32);

/// Index into [`DIRS`] for a non-zero offset.
pub fn dir_index(dx: i32, dy: i32) -> usize {
    let k = ((dy + 1) * 3 + (dx + 1)) as usize;
    if k < 4 { k } else { k - 1 }
}

/// Flag bit meaning "this tile needs its neighbour at offset (dx, dy)".
pub fn need_bit(dx: i32, dy: i32) -> u32 {
    2 << dir_index(dx, dy)
}

/// The direction opposite to `d` (`DIRS` is laid out symmetrically).
pub fn opposite(d: usize) -> usize {
    7 - d
}

/// Flags for one tile's bit-packed words ([`TILE_WORDS`] of them): [`ALIVE`], a bit
/// per neighbour it needs, and the live-cell count in bits [`POP_SHIFT`]...
/// Mirrors the logic in `sparse_life.wgsl`, for patterns placed from the CPU.
pub fn flags_for(words: &[u32]) -> u32 {
    let mut flags = 0;
    let mut pop = 0;
    for (w, &word) in words.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let i = w as u32 * 32 + bits.trailing_zeros();
            flags |= cell_flags(i % TILE, i / TILE);
            pop += 1;
            bits &= bits - 1;
        }
    }
    flags | pop << POP_SHIFT
}

fn cell_flags(x: u32, y: u32) -> u32 {
    let dx = if x < MARGIN { -1 } else if x >= TILE - MARGIN { 1 } else { 0 };
    let dy = if y < MARGIN { -1 } else if y >= TILE - MARGIN { 1 } else { 0 };
    let mut f = ALIVE;
    if dx != 0 {
        f |= need_bit(dx, 0);
    }
    if dy != 0 {
        f |= need_bit(0, dy);
    }
    if dx != 0 && dy != 0 {
        f |= need_bit(dx, dy);
    }
    f
}

/// Tile containing world cell (x, y), and the cell's index within that tile.
pub fn locate(x: i32, y: i32) -> (TileCoord, usize) {
    let t = TILE as i32;
    let tile = (x.div_euclid(t), y.div_euclid(t));
    let (lx, ly) = (x.rem_euclid(t) as usize, y.rem_euclid(t) as usize);
    (tile, ly * TILE as usize + lx)
}

/// GPU updates produced by world changes, to be applied before the next step.
#[derive(Default, Debug)]
pub struct Changes {
    /// Slots whose 8-entry neighbour row changed: (slot, row).
    pub neighbors: Vec<(u32, [u32; 8])>,
    /// Newly allocated slots, whose cells must be zeroed.
    pub new_slots: Vec<u32>,
    /// The new active-slot list, if it changed. Its order defines flag indices.
    pub active: Option<Vec<u32>>,
}

pub struct TileWorld {
    capacity: u32,
    map: HashMap<TileCoord, u32>,
    coords: Vec<TileCoord>,
    neighbors: Vec<[u32; 8]>,
    free: Vec<u32>,
    /// Allocated slots, in the order the GPU processes them (and reports flags).
    active: Vec<u32>,
    dirty_neighbors: HashSet<u32>,
    new_slots: Vec<u32>,
    active_dirty: bool,
    /// Set when a tile was needed but the pool was full (life got clipped).
    pub overflowed: bool,
}

impl TileWorld {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            coords: vec![(0, 0); capacity as usize],
            neighbors: vec![[EMPTY; 8]; capacity as usize],
            free: (0..capacity).rev().collect(),
            active: Vec::new(),
            dirty_neighbors: HashSet::new(),
            new_slots: Vec::new(),
            active_dirty: true,
            overflowed: false,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn active(&self) -> &[u32] {
        &self.active
    }

    pub fn slot(&self, coord: TileCoord) -> Option<u32> {
        self.map.get(&coord).copied()
    }

    /// Coordinates of the active tiles, in active order: what a flags readback
    /// taken now will describe. Slots can be reused before the readback is applied,
    /// so readbacks are matched to coordinates, not slots.
    pub fn active_coords(&self) -> Vec<TileCoord> {
        self.active.iter().map(|&s| self.coords[s as usize]).collect()
    }

    /// Rebuild the active list after allocations, so flags can be lined up with it.
    pub fn sync_active(&mut self) {
        if self.active_dirty {
            self.active = self.map.values().copied().collect();
        }
    }

    /// Free every tile.
    pub fn clear(&mut self) {
        let coords: Vec<TileCoord> = self.map.keys().copied().collect();
        for c in coords {
            self.release(c);
        }
        self.overflowed = false;
    }

    /// Make sure a tile exists at `coord`; returns its slot, or `None` if the pool is full.
    pub fn allocate(&mut self, coord: TileCoord) -> Option<u32> {
        if let Some(&slot) = self.map.get(&coord) {
            return Some(slot);
        }
        let Some(slot) = self.free.pop() else {
            self.overflowed = true;
            return None;
        };
        self.map.insert(coord, slot);
        self.coords[slot as usize] = coord;
        let mut row = [EMPTY; 8];
        for (d, &(dx, dy)) in DIRS.iter().enumerate() {
            if let Some(&n) = self.map.get(&(coord.0 + dx, coord.1 + dy)) {
                row[d] = n;
                self.neighbors[n as usize][opposite(d)] = slot;
                self.dirty_neighbors.insert(n);
            }
        }
        self.neighbors[slot as usize] = row;
        self.dirty_neighbors.insert(slot);
        self.new_slots.push(slot);
        self.active_dirty = true;
        Some(slot)
    }

    fn release(&mut self, coord: TileCoord) {
        let Some(slot) = self.map.remove(&coord) else { return };
        for (d, &(dx, dy)) in DIRS.iter().enumerate() {
            if let Some(&n) = self.map.get(&(coord.0 + dx, coord.1 + dy)) {
                self.neighbors[n as usize][opposite(d)] = EMPTY;
                self.dirty_neighbors.insert(n);
            }
        }
        self.neighbors[slot as usize] = [EMPTY; 8];
        self.dirty_neighbors.remove(&slot);
        self.free.push(slot);
        self.active_dirty = true;
    }

    /// Bookkeeping pass. `flags[i]` belongs to `self.active()[i]` as it was when the
    /// GPU ran. Keeps live tiles and the neighbours they need; frees everything else.
    pub fn update(&mut self, flags: &[u32]) {
        let coords = self.active_coords();
        self.update_from(&coords, flags);
    }

    /// Bookkeeping pass from a readback: `flags[i]` belongs to the tile at
    /// `coords[i]` (a snapshot from [`Self::active_coords`]). Tiles allocated since
    /// the snapshot are kept only if these flags need them, which is safe because
    /// `MARGIN >= 2 * BOOKKEEPING_INTERVAL` (see the module docs).
    pub fn update_from(&mut self, coords: &[TileCoord], flags: &[u32]) {
        debug_assert_eq!(flags.len(), coords.len());
        let mut needed: HashSet<TileCoord> = HashSet::new();
        for (&(x, y), &f) in coords.iter().zip(flags) {
            if f & ALIVE != 0 {
                needed.insert((x, y));
            }
            for &(dx, dy) in &DIRS {
                if f & need_bit(dx, dy) != 0 {
                    needed.insert((x + dx, y + dy));
                }
            }
        }

        let unneeded: Vec<TileCoord> =
            self.map.keys().filter(|c| !needed.contains(c)).copied().collect();
        for c in unneeded {
            self.release(c);
        }
        for c in needed {
            self.allocate(c);
        }
        self.sync_active();
    }

    /// Everything the GPU needs to be told since the last call.
    pub fn take_changes(&mut self) -> Changes {
        self.sync_active();
        let neighbors = self
            .dirty_neighbors
            .drain()
            .map(|s| (s, self.neighbors[s as usize]))
            .collect();
        // A slot can be freed and reused before the GPU hears about it: report each
        // still-allocated slot once.
        let mut seen = HashSet::new();
        let new_slots: Vec<u32> = std::mem::take(&mut self.new_slots)
            .into_iter()
            .filter(|&s| self.map.get(&self.coords[s as usize]) == Some(&s) && seen.insert(s))
            .collect();
        let active = std::mem::take(&mut self.active_dirty).then(|| self.active.clone());
        Changes { neighbors, new_slots, active }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dir_index_matches_dirs_and_opposites() {
        for (d, &(dx, dy)) in DIRS.iter().enumerate() {
            assert_eq!(dir_index(dx, dy), d);
            assert_eq!(DIRS[opposite(d)], (-dx, -dy));
        }
    }

    #[test]
    fn locate_handles_negative_coordinates() {
        assert_eq!(locate(0, 0), ((0, 0), 0));
        assert_eq!(locate(-1, -1), ((-1, -1), TILE_CELLS - 1));
        let t = TILE as i32;
        assert_eq!(locate(t, 1), ((1, 0), TILE as usize));
        assert_eq!(locate(-t, 0), ((-1, 0), 0));
        assert_eq!(locate(-1, 0), ((-1, 0), TILE as usize - 1));
    }

    #[test]
    fn cell_flags_regions() {
        // Corners need their two edge neighbours and the diagonal one.
        assert_eq!(cell_flags(0, 0), ALIVE | need_bit(-1, 0) | need_bit(0, -1) | need_bit(-1, -1));
        assert_eq!(cell_flags(TILE - 1, TILE - 1), ALIVE | need_bit(1, 0) | need_bit(0, 1) | need_bit(1, 1));
        // Just inside the margin on the east side, just outside it on the south.
        let (x, y) = (TILE - MARGIN, TILE - MARGIN - 1);
        let expected_y = if y < MARGIN { need_bit(0, -1) | need_bit(1, -1) } else { 0 };
        assert_eq!(cell_flags(x, y), ALIVE | need_bit(1, 0) | expected_y);
    }

    #[test]
    fn flags_for_reads_packed_bits() {
        let mut words = vec![0u32; TILE_WORDS];
        let mut set = |x: i32, y: i32| {
            let (_, i) = locate(x, y);
            words[i / 32] |= 1 << (i % 32);
        };
        // Cells in different words and rows; flags_for must match the per-cell flags.
        let cells = [(0, 0), (33, 5), (63, 63), (31, 40)];
        for &(x, y) in &cells {
            set(x, y);
        }
        let f = flags_for(&words);
        assert_eq!(f >> POP_SHIFT, cells.len() as u32);
        let expected = cells.iter().fold(0, |acc, &(x, y)| acc | cell_flags(x as u32, y as u32));
        assert_eq!(f & ((1 << POP_SHIFT) - 1), expected);
    }

    #[test]
    fn allocate_links_neighbours_both_ways() {
        let mut w = TileWorld::new(8);
        let a = w.allocate((0, 0)).unwrap();
        let b = w.allocate((1, 0)).unwrap();
        assert_eq!(w.neighbors[a as usize][dir_index(1, 0)], b);
        assert_eq!(w.neighbors[b as usize][dir_index(-1, 0)], a);
        w.release((1, 0));
        assert_eq!(w.neighbors[a as usize][dir_index(1, 0)], EMPTY);
    }

    #[test]
    fn update_keeps_needed_and_frees_rest() {
        let mut w = TileWorld::new(16);
        w.allocate((0, 0));
        w.allocate((5, 5));
        w.take_changes();
        // (0,0) alive near its east edge; (5,5) empty.
        let flags: Vec<u32> = w
            .active()
            .iter()
            .map(|&s| if w.coords[s as usize] == (0, 0) { ALIVE | need_bit(1, 0) } else { 0 })
            .collect();
        w.update(&flags);
        assert!(w.slot((0, 0)).is_some());
        assert!(w.slot((1, 0)).is_some());
        assert!(w.slot((5, 5)).is_none());
        assert_eq!(w.len(), 2);
        let ch = w.take_changes();
        assert_eq!(ch.new_slots, vec![w.slot((1, 0)).unwrap()]);
        assert_eq!(ch.active.unwrap().len(), 2);
    }

    #[test]
    fn pool_overflow_is_reported() {
        let mut w = TileWorld::new(1);
        assert!(w.allocate((0, 0)).is_some());
        assert!(w.allocate((1, 0)).is_none());
        assert!(w.overflowed);
    }

    #[test]
    fn reused_slot_is_zeroed_once() {
        let mut w = TileWorld::new(4);
        let s = w.allocate((0, 0)).unwrap();
        w.release((0, 0));
        let s2 = w.allocate((3, 3)).unwrap();
        assert_eq!(s, s2);
        assert_eq!(w.take_changes().new_slots, vec![s2]);
    }
}
