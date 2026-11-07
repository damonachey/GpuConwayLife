// Conway's Game of Life on a sparse, tiled world, entirely on the GPU.
//
// The world is made of TILE x TILE tiles stored in a pool (`src` / `dst`,
// ping-ponged each generation). Cells are bit-packed: a tile is TILE rows of
// WORDS_PER_ROW u32 words, and bit i of word wx in row y is tile-local cell
// (wx * 32 + i, y). One workgroup processes one active tile.
//
// Each workgroup first copies its tile plus a one-word / one-row border (taken from
// the neighbour tiles via the neighbour table; a missing neighbour (EMPTY) reads as
// dead) into workgroup memory, then computes every word from there. The CPU
// guarantees missing neighbours can't matter until its next bookkeeping pass (see
// tiles.rs).
//
// Each workgroup also reports flags for its tile: bit 0 = any live cell, bit 1+d =
// "a live cell is within MARGIN of the edge/corner toward neighbour d", and bits
// 16.. = the tile's live-cell count (for the population readout).
//
// TILE and MARGIN must match tiles.rs.

const TILE: u32 = 128u;
const MARGIN: u32 = 64u;
const WORDS_PER_ROW: u32 = TILE / 32u;
const TILE_WORDS: u32 = TILE * WORDS_PER_ROW;
const THREADS: u32 = 256u;
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

// Per-bit counter: s.x = ones, s.y = twos, s.z = "four or more" (sticky).
// That's all Life needs: alive next iff count == 3, or count == 2 and alive now.
fn add_bit(s: ptr<function, vec3<u32>>, x: u32) {
    let c0 = (*s).x & x;
    (*s).x ^= x;
    let c1 = (*s).y & c0;
    (*s).y ^= c0;
    (*s).z |= c1;
}

@compute @workgroup_size(256)
fn step(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lindex: u32,
) {
    // Tiles are spread over a 2D dispatch to stay under the 65535 per-axis limit.
    let index = wid.y * nwg.x + wid.x;
    if (index >= params.count) {
        return; // uniform per workgroup, so the barriers below stay valid
    }
    let slot = active_tiles[index];

    if (lindex == 0u) {
        atomicStore(&tile_flags, 0u);
        atomicStore(&tile_pop, 0u);
    }
    if (lindex < 8u) {
        near_slots[lindex] = neighbors[slot * 8u + lindex];
    }
    workgroupBarrier();

    // Copy the tile and its border into workgroup memory.
    for (var i = lindex; i < HALO_WORDS; i += THREADS) {
        let wx = i32(i % HALO_COLS) - 1;
        let y = i32(i / HALO_COLS) - 1;
        halo[i] = load_word(slot, wx, y);
    }
    workgroupBarrier();

    var f = 0u;
    var pop = 0u;
    for (var word = lindex; word < TILE_WORDS; word += THREADS) {
        let wx = word % WORDS_PER_ROW;
        let y = word / WORDS_PER_ROW;
        // Centre of this word in the halo; neighbours are +-1 column, +-HALO_COLS.
        let c = (y + 1u) * HALO_COLS + wx + 1u;
        let ul = halo[c - HALO_COLS - 1u];
        let u = halo[c - HALO_COLS];
        let ur = halo[c - HALO_COLS + 1u];
        let ml = halo[c - 1u];
        let m = halo[c];
        let mr = halo[c + 1u];
        let dl = halo[c + HALO_COLS - 1u];
        let d = halo[c + HALO_COLS];
        let dr = halo[c + HALO_COLS + 1u];

        // West neighbour of bit i is bit i-1 (from the left word at bit 0);
        // east neighbour is bit i+1 (from the right word at bit 31).
        var s = vec3<u32>(0u);
        add_bit(&s, (u << 1u) | (ul >> 31u));
        add_bit(&s, u);
        add_bit(&s, (u >> 1u) | (ur << 31u));
        add_bit(&s, (m << 1u) | (ml >> 31u));
        add_bit(&s, (m >> 1u) | (mr << 31u));
        add_bit(&s, (d << 1u) | (dl >> 31u));
        add_bit(&s, d);
        add_bit(&s, (d >> 1u) | (dr << 31u));
        // count 2 -> (ones=0, twos=1), count 3 -> (ones=1, twos=1), and fewer than 4.
        let next = s.y & ~s.z & (s.x | m);
        dst[slot * TILE_WORDS + word] = next;

        if (next != 0u) {
            // Which edges / corners have a live cell within MARGIN of them.
            let near_w = (next & bits_in(wx, 0u, MARGIN)) != 0u;
            let near_e = (next & bits_in(wx, TILE - MARGIN, TILE)) != 0u;
            let near_n = y < MARGIN;
            let near_s = y >= TILE - MARGIN;
            f |= 1u;
            if (near_w) { f |= 2u << dir_index(-1, 0); }
            if (near_e) { f |= 2u << dir_index(1, 0); }
            if (near_n) { f |= 2u << dir_index(0, -1); }
            if (near_s) { f |= 2u << dir_index(0, 1); }
            if (near_w && near_n) { f |= 2u << dir_index(-1, -1); }
            if (near_e && near_n) { f |= 2u << dir_index(1, -1); }
            if (near_w && near_s) { f |= 2u << dir_index(-1, 1); }
            if (near_e && near_s) { f |= 2u << dir_index(1, 1); }
            pop += countOneBits(next);
        }
    }
    if (f != 0u) {
        atomicOr(&tile_flags, f);
        atomicAdd(&tile_pop, pop);
    }
    workgroupBarrier();
    if (lindex == 0u) {
        flags[index] = atomicLoad(&tile_flags) | (atomicLoad(&tile_pop) << 16u);
    }
}
