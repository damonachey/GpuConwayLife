// Conway's Game of Life on a sparse, tiled world, entirely on the GPU.
//
// The world is made of 64x64 tiles stored in a pool (`src` / `dst`, one u32 per
// cell, ping-ponged each generation). One workgroup processes one active tile.
// Cells outside a tile are read through that tile's neighbour table; a missing
// neighbour (EMPTY) counts as dead. The CPU guarantees that can't matter for the
// next MARGIN generations (see tiles.rs).
//
// Each workgroup also reports flags for its tile: bit 0 = any live cell, bit 1+d =
// "a live cell is within MARGIN of the edge/corner toward neighbour d", and bits
// 16..28 = the tile's live-cell count (for the population readout).

const TILE: u32 = 64u;
const TILE_CELLS: u32 = 4096u;
const MARGIN: u32 = 16u;
const EMPTY: u32 = 0xffffffffu;
const BLOCK: u32 = 4u; // each thread computes a 4x4 block of cells

struct Params {
    count: u32, // number of active tiles
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> active_tiles: array<u32>; // slot per active index
@group(0) @binding(2) var<storage, read> neighbors: array<u32>;   // 8 slots per slot
@group(0) @binding(3) var<storage, read> src: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst: array<u32>;
@group(0) @binding(5) var<storage, read_write> flags: array<u32>; // per active index

var<workgroup> tile_flags: atomic<u32>;
var<workgroup> tile_pop: atomic<u32>;

// Index into the neighbour row for a non-zero offset; matches DIRS in tiles.rs.
fn dir_index(dx: i32, dy: i32) -> u32 {
    let k = u32((dy + 1) * 3 + (dx + 1));
    return select(k - 1u, k, k < 4u);
}

// Cell at tile-local (x, y), where x and y may be one step outside the tile.
fn read_cell(slot: u32, x: i32, y: i32) -> u32 {
    let t = i32(TILE);
    let dx = select(select(0, 1, x >= t), -1, x < 0);
    let dy = select(select(0, 1, y >= t), -1, y < 0);
    var s = slot;
    if (dx != 0 || dy != 0) {
        s = neighbors[slot * 8u + dir_index(dx, dy)];
        if (s == EMPTY) {
            return 0u;
        }
    }
    let lx = u32(x - dx * t);
    let ly = u32(y - dy * t);
    return src[s * TILE_CELLS + ly * TILE + lx];
}

fn cell_flags(x: u32, y: u32) -> u32 {
    let dx = select(select(0, 1, x >= TILE - MARGIN), -1, x < MARGIN);
    let dy = select(select(0, 1, y >= TILE - MARGIN), -1, y < MARGIN);
    var f = 1u;
    if (dx != 0) {
        f |= 2u << dir_index(dx, 0);
    }
    if (dy != 0) {
        f |= 2u << dir_index(0, dy);
    }
    if (dx != 0 && dy != 0) {
        f |= 2u << dir_index(dx, dy);
    }
    return f;
}

@compute @workgroup_size(16, 16)
fn step(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
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

    // Load this thread's 4x4 block plus a 1-cell border.
    let x0 = i32(lid.x * BLOCK);
    let y0 = i32(lid.y * BLOCK);
    var win: array<u32, 36>;
    for (var j = 0; j < 6; j++) {
        for (var i = 0; i < 6; i++) {
            win[j * 6 + i] = read_cell(slot, x0 - 1 + i, y0 - 1 + j);
        }
    }

    var f = 0u;
    var pop = 0u;
    for (var j = 1; j <= 4; j++) {
        for (var i = 1; i <= 4; i++) {
            let n = win[(j - 1) * 6 + i - 1] + win[(j - 1) * 6 + i] + win[(j - 1) * 6 + i + 1]
                  + win[j * 6 + i - 1]                           + win[j * 6 + i + 1]
                  + win[(j + 1) * 6 + i - 1] + win[(j + 1) * 6 + i] + win[(j + 1) * 6 + i + 1];
            let alive = win[j * 6 + i];
            let next = select(0u, 1u, n == 3u || (alive == 1u && n == 2u));
            let x = u32(x0 + i - 1);
            let y = u32(y0 + j - 1);
            dst[slot * TILE_CELLS + y * TILE + x] = next;
            if (next == 1u) {
                f |= cell_flags(x, y);
                pop += 1u;
            }
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
