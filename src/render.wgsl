// Draws the grid at native resolution: one cell = one screen pixel.
// One full-screen triangle; the fragment shader maps each pixel to a cell.
// The grid is centred in the window; outside it is a darker border.

struct View {
    grid: vec2<u32>,   // grid size in cells
    offset: vec2<i32>, // screen pixel where cell (0, 0) is drawn (may be negative)
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> cells: array<u32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Three vertices that cover the screen: (-1,-1), (3,-1), (-1,3).
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let border_color = vec3<f32>(0.0, 0.0, 0.0);
    let dead_color = vec3<f32>(0.01, 0.012, 0.02);
    let live_color = vec3<f32>(0.35, 0.95, 0.55);

    let c = vec2<i32>(floor(frag.xy)) - view.offset;
    if (c.x < 0 || c.y < 0 || c.x >= i32(view.grid.x) || c.y >= i32(view.grid.y)) {
        return vec4<f32>(border_color, 1.0);
    }
    let alive = cells[u32(c.y) * view.grid.x + u32(c.x)];
    return vec4<f32>(select(dead_color, live_color, alive == 1u), 1.0);
}
