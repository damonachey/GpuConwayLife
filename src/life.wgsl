// Conway's Game of Life, entirely on the GPU.
//
// The grid lives in two storage buffers (one u32 per cell, 0 = dead, 1 = alive).
// Each frame the compute pass reads `src` and writes the next generation to `dst`,
// then the CPU swaps which buffer is which (ping-pong). The render pass reads the
// newest buffer directly, so cell data never goes back to the CPU.

struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn cell(x: i32, y: i32) -> u32 {
    // Wrap around the edges, so the grid is a torus.
    let w = i32(params.width);
    let h = i32(params.height);
    let wx = (x + w) % w;
    let wy = (y + h) % h;
    return src[u32(wy) * params.width + u32(wx)];
}

@compute @workgroup_size(8, 8)
fn step(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);

    let n = cell(x - 1, y - 1) + cell(x, y - 1) + cell(x + 1, y - 1)
          + cell(x - 1, y)                      + cell(x + 1, y)
          + cell(x - 1, y + 1) + cell(x, y + 1) + cell(x + 1, y + 1);

    let alive = cell(x, y);
    var next = 0u;
    if (n == 3u || (alive == 1u && n == 2u)) {
        next = 1u;
    }
    dst[id.y * params.width + id.x] = next;
}
