// Conway's Game of Life, entirely on the GPU, with cells packed 32 to a u32.
//
// Bit i of word (wx, y) is cell (wx * 32 + i, y). Beyond the edges everything is
// dead, so patterns that reach the border die there rather than wrapping.
// Each frame the compute pass reads `src` and writes the next generation to `dst`,
// then the CPU swaps which buffer is which (ping-pong).
//
// One thread computes one whole word (32 cells) at once with `next_word` from
// life_rule.wgsl, which Rust prepends to this file.

struct Params {
    width: u32,         // cells per row (a multiple of 32)
    height: u32,        // rows
    words_per_row: u32, // width / 32
    seed: u32,          // for the `seed` entry point
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64, 4)
fn step(@builtin(global_invocation_id) id: vec3<u32>) {
    let wpr = params.words_per_row;
    let h = params.height;
    if (id.x >= wpr || id.y >= h) {
        return;
    }
    // Everything beyond the grid's edge is permanently dead: off-grid neighbour
    // words read as 0, so cells that leave the grid die instead of wrapping.
    let has_l = id.x > 0u;
    let has_r = id.x + 1u < wpr;
    let has_u = id.y > 0u;
    let has_d = id.y + 1u < h;
    let rm = id.y * wpr;
    let ru = rm - wpr; // only used when has_u
    let rd = rm + wpr; // only used when has_d

    var ul = 0u; var u = 0u; var ur = 0u;
    var ml = 0u;             var mr = 0u;
    var dl = 0u; var d = 0u; var dr = 0u;
    let m = src[rm + id.x];
    if (has_l) { ml = src[rm + id.x - 1u]; }
    if (has_r) { mr = src[rm + id.x + 1u]; }
    if (has_u) {
        u = src[ru + id.x];
        if (has_l) { ul = src[ru + id.x - 1u]; }
        if (has_r) { ur = src[ru + id.x + 1u]; }
    }
    if (has_d) {
        d = src[rd + id.x];
        if (has_l) { dl = src[rd + id.x - 1u]; }
        if (has_r) { dr = src[rd + id.x + 1u]; }
    }

    dst[rm + id.x] = next_word(ul, u, ur, ml, m, mr, dl, d, dr);
}

// Population count of `src`. Workgroup totals are spread over COUNTERS slots so no
// single u32 can overflow, even for a ~17 billion cell grid; the CPU adds them up.
const COUNTERS: u32 = 1024u;
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>, COUNTERS>;
var<workgroup> group_count: atomic<u32>;

@compute @workgroup_size(64, 4)
fn count(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) thread: u32,
) {
    if (thread == 0u) {
        atomicStore(&group_count, 0u);
    }
    workgroupBarrier();
    if (id.x < params.words_per_row && id.y < params.height) {
        let bits = countOneBits(src[id.y * params.words_per_row + id.x]);
        if (bits != 0u) {
            atomicAdd(&group_count, bits);
        }
    }
    workgroupBarrier();
    if (thread == 0u) {
        atomicAdd(&counts[(wid.y * 7u + wid.x) % COUNTERS], atomicLoad(&group_count));
    }
}

// PCG-style integer hash.
fn hash(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// Fill `dst` with random soup: each bit is the AND of two random bits, so ~25% alive.
@compute @workgroup_size(64, 4)
fn seed(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.words_per_row || id.y >= params.height) {
        return;
    }
    let i = id.y * params.words_per_row + id.x;
    let a = hash(i ^ hash(params.seed));
    let b = hash(a ^ 0x9e3779b9u);
    dst[i] = a & b;
}
