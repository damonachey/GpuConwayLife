// Conway's Game of Life on a sparse, tiled world, entirely on the GPU.
//
// The world is made of 64x64 tiles stored in a pool (`src` / `dst`, ping-ponged
// each generation). Cells are bit-packed: a tile is 64 rows of 2 u32 words, and bit
// i of word wx in row y is tile-local cell (wx * 32 + i, y). One 128-thread
// workgroup processes one active tile, one thread per word.
//
// Words outside a tile are read through that tile's neighbour table; a missing
// neighbour (EMPTY) counts as dead. The CPU guarantees that can't matter for the
// next MARGIN generations (see tiles.rs).
//
// Each workgroup also reports flags for its tile: bit 0 = any live cell, bit 1+d =
// "a live cell is within MARGIN of the edge/corner toward neighbour d", and bits
// 16..28 = the tile's live-cell count (for the population readout).

const TILE: u32 = 64u;
const WORDS_PER_ROW: u32 = 2u;
const TILE_WORDS: u32 = 128u;
const MARGIN: u32 = 16u;
const EMPTY: u32 = 0xffffffffu;

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

// Index into the neighbour row for a non-zero offset; matches DIRS in tiles.rs.
fn dir_index(dx: i32, dy: i32) -> u32 {
    let k = u32((dy + 1) * 3 + (dx + 1));
    return select(k - 1u, k, k < 4u);
}

// Word at tile-local word column `wx` (-1..2) and row `y` (-1..64): may be one word
// or one row outside the tile, in which case it comes from a neighbour tile.
fn read_word(slot: u32, wx: i32, y: i32) -> u32 {
    let wpr = i32(WORDS_PER_ROW);
    let t = i32(TILE);
    let dx = select(select(0, 1, wx >= wpr), -1, wx < 0);
    let dy = select(select(0, 1, y >= t), -1, y < 0);
    var s = slot;
    if (dx != 0 || dy != 0) {
        s = neighbors[slot * 8u + dir_index(dx, dy)];
        if (s == EMPTY) {
            return 0u;
        }
    }
    let lx = u32(wx - dx * wpr);
    let ly = u32(y - dy * t);
    return src[s * TILE_WORDS + ly * WORDS_PER_ROW + lx];
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

@compute @workgroup_size(128)
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
    workgroupBarrier();

    let wx = i32(lindex % WORDS_PER_ROW);
    let y = i32(lindex / WORDS_PER_ROW);
    let ul = read_word(slot, wx - 1, y - 1);
    let u = read_word(slot, wx, y - 1);
    let ur = read_word(slot, wx + 1, y - 1);
    let ml = read_word(slot, wx - 1, y);
    let m = src[slot * TILE_WORDS + lindex];
    let mr = read_word(slot, wx + 1, y);
    let dl = read_word(slot, wx - 1, y + 1);
    let d = read_word(slot, wx, y + 1);
    let dr = read_word(slot, wx + 1, y + 1);

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
    dst[slot * TILE_WORDS + lindex] = next;

    if (next != 0u) {
        // Which edges / corners have a live cell within MARGIN of them. Word 0
        // holds x = 0..31 (west margin = bits 0..15); word 1 holds x = 32..63
        // (east margin = bits 16..31). All bits share row y.
        let near_w = wx == 0 && (next & 0xffffu) != 0u;
        let near_e = wx == 1 && (next & 0xffff0000u) != 0u;
        let near_n = u32(y) < MARGIN;
        let near_s = u32(y) >= TILE - MARGIN;
        var f = 1u;
        if (near_w) { f |= 2u << dir_index(-1, 0); }
        if (near_e) { f |= 2u << dir_index(1, 0); }
        if (near_n) { f |= 2u << dir_index(0, -1); }
        if (near_s) { f |= 2u << dir_index(0, 1); }
        if (near_w && near_n) { f |= 2u << dir_index(-1, -1); }
        if (near_e && near_n) { f |= 2u << dir_index(1, -1); }
        if (near_w && near_s) { f |= 2u << dir_index(-1, 1); }
        if (near_e && near_s) { f |= 2u << dir_index(1, 1); }
        atomicOr(&tile_flags, f);
        atomicAdd(&tile_pop, countOneBits(next));
    }
    workgroupBarrier();
    if (lindex == 0u) {
        flags[index] = atomicLoad(&tile_flags) | (atomicLoad(&tile_pop) << 16u);
    }
}
