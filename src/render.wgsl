// Draws the grid by stretching it over the whole window.
// One full-screen triangle; the fragment shader looks up which cell each pixel is in.

struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<u32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    // Three vertices that cover the screen: (-1,-1), (3,-1), (-1,3).
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: VsOut;
    out.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let x = min(u32(in.uv.x * f32(params.width)), params.width - 1u);
    let y = min(u32(in.uv.y * f32(params.height)), params.height - 1u);
    let alive = cells[y * params.width + x];
    let dead_color = vec3<f32>(0.04, 0.05, 0.08);
    let live_color = vec3<f32>(0.35, 0.95, 0.55);
    return vec4<f32>(select(dead_color, live_color, alive == 1u), 1.0);
}
