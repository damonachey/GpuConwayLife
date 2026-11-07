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
//! Then the world can safely run `MARGIN` generations before we look again: no live
//! cell can reach a tile that isn't allocated. The compute shader reports, per tile,
//! whether it is alive and which neighbours it needs (see [`flags_for`]).

use std::collections::{HashMap, HashSet};

/// Tile edge length in cells.
pub const TILE: u32 = 64;
pub const TILE_CELLS: usize = (TILE * TILE) as usize;
/// Distance from a tile edge that counts as "near" it, and therefore the number of
/// generations that may run between bookkeeping passes. Must be <= TILE / 2.
pub const MARGIN: u32 = 16;
/// Neighbour-table entry for "no tile here".
pub const EMPTY: u32 = u32::MAX;
/// Flag bit: the tile has at least one live cell.
pub const ALIVE: u32 = 1;
/// Flags bits 16.. hold the tile's live-cell count.
pub const POP_SHIFT: u32 = 16;
/// Neighbour directions as (dx, dy), y pointing down. Flag bit `1 << (1 + d)` means
/// "tile needs neighbour `d`". Note `DIRS[7 - d]` is the opposite of `DIRS[d]`.
pub const DIRS: [(i32, i32); 8] =
    [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

pub type TileCoord = (i32, i32);

/// Index into [`DIRS`] for a non-zero offset.
pub fn dir_index(dx: i32, dy: i32) -> usize {
    let k = ((dy + 1) * 3 + (dx + 1)) as usize;
    if k < 4 { k } else { k - 1 }
}

/// Flags for one tile's cells: [`ALIVE`], a bit per neighbour it needs, and the
/// live-cell count in bits [`POP_SHIFT`]...
/// Mirrors the logic in `life.wgsl`, for patterns placed from the CPU.
pub fn flags_for(cells: &[u32]) -> u32 {
    let mut flags = 0;
    let mut pop = 0;
    for (i, &alive) in cells.iter().enumerate() {
        if alive != 0 {
            flags |= cell_flags(i as u32 % TILE, i as u32 / TILE);
            pop += 1;
        }
    }
    flags | pop << POP_SHIFT
}

fn cell_flags(x: u32, y: u32) -> u32 {
    let dx = if x < MARGIN { -1 } else if x >= TILE - MARGIN { 1 } else { 0 };
    let dy = if y < MARGIN { -1 } else if y >= TILE - MARGIN { 1 } else { 0 };
    let mut f = ALIVE;
    if dx != 0 {
        f |= 2 << dir_index(dx, 0);
    }
    if dy != 0 {
        f |= 2 << dir_index(0, dy);
    }
    if dx != 0 && dy != 0 {
        f |= 2 << dir_index(dx, dy);
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
                self.neighbors[n as usize][7 - d] = slot;
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
                self.neighbors[n as usize][7 - d] = EMPTY;
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
        debug_assert_eq!(flags.len(), self.active.len());
        let mut needed: HashSet<TileCoord> = HashSet::new();
        for (&slot, &f) in self.active.iter().zip(flags) {
            let (x, y) = self.coords[slot as usize];
            if f & ALIVE != 0 {
                needed.insert((x, y));
            }
            for (d, &(dx, dy)) in DIRS.iter().enumerate() {
                if f & (2 << d) != 0 {
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
            assert_eq!(DIRS[7 - d], (-dx, -dy));
        }
    }

    #[test]
    fn locate_handles_negative_coordinates() {
        assert_eq!(locate(0, 0), ((0, 0), 0));
        assert_eq!(locate(-1, -1), ((-1, -1), TILE_CELLS - 1));
        assert_eq!(locate(64, 1), ((1, 0), 64));
        assert_eq!(locate(-64, 0), ((-1, 0), 0));
    }

    #[test]
    fn cell_flags_regions() {
        let mid = TILE / 2;
        assert_eq!(cell_flags(mid, mid), ALIVE);
        // Near west edge only: needs west neighbour.
        assert_eq!(cell_flags(0, mid), ALIVE | 2 << dir_index(-1, 0));
        // Near south-east corner: needs east, south and south-east.
        let se = cell_flags(TILE - 1, TILE - 1);
        assert_eq!(
            se,
            ALIVE | 2 << dir_index(1, 0) | 2 << dir_index(0, 1) | 2 << dir_index(1, 1)
        );
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
            .map(|&s| if w.coords[s as usize] == (0, 0) { ALIVE | 2 << dir_index(1, 0) } else { 0 })
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
