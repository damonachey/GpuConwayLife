// Conway's Game of Life, entirely on the GPU, with cells packed 32 to a u32.
//
// Bit i of word (wx, y) is cell (wx * 32 + i, y). Beyond the edges everything is
// dead, so patterns that reach the border die there rather than wrapping.
// Each frame the compute pass reads `src` and writes the next generation to `dst`,
// then the CPU swaps which buffer is which (ping-pong).
//
// One thread computes one whole word (32 cells) at once: it shifts the nine
// surrounding words into eight "neighbour" words, then adds them with a bitwise
// adder, so every bit position gets its own neighbour count in parallel.

struct Params {
    width: u32,         // cells per row (a multiple of 32)
    height: u32,        // rows
    words_per_row: u32, // width / 32
    seed: u32,          // for the `seed` entry point
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

// Per-bit counter: s.x = ones, s.y = twos, s.z = "four or more" (sticky).
// That's all Life needs: alive next iff count == 3, or count == 2 and alive now.
fn add_bit(s: ptr<function, vec3<u32>>, x: u32) {
    let c0 = (*s).x & x;
    (*s).x ^= x;
    let c1 = (*s).y & c0;
    (*s).y ^= c0;
    (*s).z |= c1;
}

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
    dst[rm + id.x] = s.y & ~s.z & (s.x | m);
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
    @builtin(local_invocation_index) lindex: u32,
) {
    if (lindex == 0u) {
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
    if (lindex == 0u) {
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
