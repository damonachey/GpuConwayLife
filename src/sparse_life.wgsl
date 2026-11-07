// Conway's Game of Life on a sparse, tiled world, entirely on the GPU.
//
// The world is made of TILE x TILE tiles stored in a pool (`src` / `dst`,
// ping-ponged each generation). Cells are bit-packed: a tile is TILE rows of
// WORDS_PER_ROW u32 words, and bit i of word wx in row y is tile-local cell
// (wx * 32 + i, y). One workgroup processes one active tile.
//
// Each workgroup first copies its tile plus a one-word / one-row border (taken from
// the neighbour tiles via the neighbour table; a missing neighbour (EMPTY) reads as
// dead) into workgroup memory, then computes every word from there with
// `next_word` (life_rule.wgsl). The CPU guarantees missing neighbours can't matter
// until its next bookkeeping pass (see tiles.rs).
//
// Each workgroup also reports flags for its tile: bit 0 = any live cell, bit 1+d =
// "a live cell is within MARGIN of the edge/corner toward neighbour d", and bits
// POP_SHIFT.. = the tile's live-cell count (for the population readout).
//
// TILE, MARGIN and POP_SHIFT are prepended by Rust from tiles.rs, along with
// life_rule.wgsl.

const WORDS_PER_ROW: u32 = TILE / 32u;
const TILE_WORDS: u32 = TILE * WORDS_PER_ROW;
// One thread per word, up to 256 (the most every wgpu device must allow).
const THREADS: u32 = min(TILE_WORDS, 256u);
const EMPTY: u32 = 0xffffffffu;
// The tile plus a border: rows -1..TILE, word columns -1..WORDS_PER_ROW.
const HALO_COLS: u32 = WORDS_PER_ROW + 2u;
const HALO_ROWS: u32 = TILE + 2u;
const HALO_WORDS: u32 = HALO_COLS * HALO_ROWS;

struct Params {
    count: u32, // number of active tiles
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> active_tiles: array<u32>; // slot per active index
@group(0) @binding(2) var<storage, read> neighbors: array<u32>;    // 8 slots per slot
@group(0) @binding(3) var<storage, read> src: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst: array<u32>;
@group(0) @binding(5) var<storage, read_write> flags: array<u32>;  // per active index

var<workgroup> tile_flags: atomic<u32>;
var<workgroup> tile_pop: atomic<u32>;
/// This tile's 8 neighbour slots (DIRS order).
var<workgroup> near_slots: array<u32, 8>;
/// The tile plus its border, as `halo[(y + 1) * HALO_COLS + (wx + 1)]`.
var<workgroup> halo: array<u32, HALO_WORDS>;

// Index into the neighbour row for a non-zero offset; matches DIRS in tiles.rs.
fn dir_index(dx: i32, dy: i32) -> u32 {
    let k = u32((dy + 1) * 3 + (dx + 1));
    return select(k - 1u, k, k < 4u);
}

// Flag bit meaning "this tile needs its neighbour at offset (dx, dy)".
fn need_bit(dx: i32, dy: i32) -> u32 {
    return 2u << dir_index(dx, dy);
}

// Word at tile-local word column `wx` (-1..WORDS_PER_ROW) and row `y` (-1..TILE),
// from this tile or a neighbour, straight from the pool.
fn load_word(slot: u32, wx: i32, y: i32) -> u32 {
    let wpr = i32(WORDS_PER_ROW);
    let t = i32(TILE);
    let dx = select(select(0, 1, wx >= wpr), -1, wx < 0);
    let dy = select(select(0, 1, y >= t), -1, y < 0);
    var s = slot;
    if (dx != 0 || dy != 0) {
        s = near_slots[dir_index(dx, dy)];
        if (s == EMPTY) {
            return 0u;
        }
    }
    let lx = u32(wx - dx * wpr);
    let ly = u32(y - dy * t);
    return src[s * TILE_WORDS + ly * WORDS_PER_ROW + lx];
}

// Bits of word column `wx` whose cells lie in x = [lo, hi).
fn bits_in(wx: u32, lo: u32, hi: u32) -> u32 {
    let start = max(lo, wx * 32u);
    let end = min(hi, wx * 32u + 32u);
    if (start >= end) {
        return 0u;
    }
    let n = end - start;
    let ones = select((1u << n) - 1u, 0xffffffffu, n == 32u);
    return ones << (start - wx * 32u);
}

// Flags for a non-empty word `next` at word column `wx`, row `y`: the same as
// `cell_flags` in tiles.rs, for all of the word's live cells at once. (MARGIN is a
// multiple of 32, so a word is never near both the west and east edges.)
fn word_flags(next: u32, wx: u32, y: u32) -> u32 {
    let near_w = (next & bits_in(wx, 0u, MARGIN)) != 0u;
    let near_e = (next & bits_in(wx, TILE - MARGIN, TILE)) != 0u;
    let dx = select(select(0, 1, near_e), -1, near_w);
    let dy = select(select(0, 1, y >= TILE - MARGIN), -1, y < MARGIN);
    var f = 1u; // alive
    if (dx != 0) { f |= need_bit(dx, 0); }
    if (dy != 0) { f |= need_bit(0, dy); }
    if (dx != 0 && dy != 0) { f |= need_bit(dx, dy); }
    return f;
}

@compute @workgroup_size(THREADS)
fn step(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) thread: u32,
) {
    // Tiles are spread over a 2D dispatch to stay under the 65535 per-axis limit.
    let index = wid.y * nwg.x + wid.x;
    if (index >= params.count) {
        return; // uniform per workgroup, so the barriers below stay valid
    }
    let slot = active_tiles[index];

    if (thread == 0u) {
        atomicStore(&tile_flags, 0u);
        atomicStore(&tile_pop, 0u);
    }
    if (thread < 8u) {
        near_slots[thread] = neighbors[slot * 8u + thread];
    }
    workgroupBarrier();

    // Copy the tile and its border into workgroup memory.
    for (var i = thread; i < HALO_WORDS; i += THREADS) {
        let wx = i32(i % HALO_COLS) - 1;
        let y = i32(i / HALO_COLS) - 1;
        halo[i] = load_word(slot, wx, y);
    }
    workgroupBarrier();

    var f = 0u;
    var pop = 0u;
    for (var word = thread; word < TILE_WORDS; word += THREADS) {
        let wx = word % WORDS_PER_ROW;
        let y = word / WORDS_PER_ROW;
        // Centre of this word in the halo; neighbours are +-1 column, +-HALO_COLS.
        let c = (y + 1u) * HALO_COLS + wx + 1u;
        let up = c - HALO_COLS;
        let down = c + HALO_COLS;
        let next = next_word(
            halo[up - 1u], halo[up], halo[up + 1u],
            halo[c - 1u], halo[c], halo[c + 1u],
            halo[down - 1u], halo[down], halo[down + 1u],
        );
        dst[slot * TILE_WORDS + word] = next;
        if (next != 0u) {
            f |= word_flags(next, wx, y);
            pop += countOneBits(next);
        }
    }
    if (f != 0u) {
        atomicOr(&tile_flags, f);
        atomicAdd(&tile_pop, pop);
    }
    workgroupBarrier();
    if (thread == 0u) {
        flags[index] = atomicLoad(&tile_flags) | (atomicLoad(&tile_pop) << POP_SHIFT);
    }
}
