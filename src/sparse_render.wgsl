// Draws the tiled world at native resolution: one cell = one screen pixel.
// One full-screen triangle; each pixel finds its tile through a small map of the
// tiles currently on screen (built by the CPU), then reads the cell's bit from the
// pool (TILE rows of TILE / 32 words per tile; bit i of word wx is cell
// x = wx * 32 + i).
//
// TILE and TILE_SHIFT (log2 of TILE) are prepended by Rust from tiles.rs.

const WORDS_PER_ROW: u32 = TILE / 32u;
const TILE_WORDS: u32 = TILE * WORDS_PER_ROW;
const EMPTY: u32 = 0xffffffffu;

struct View {
    screen_origin: vec2<i32>, // world cell shown at pixel (0, 0)
    map_origin: vec2<i32>,    // tile coordinate of map entry (0, 0)
    map_size: vec2<u32>,      // map columns, rows
    show_tiles: u32,          // 1 = shade allocated tiles
    _pad: u32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> tile_map: array<u32>; // slot or EMPTY
@group(0) @binding(2) var<storage, read> cells: array<u32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Three vertices that cover the screen: (-1,-1), (3,-1), (-1,3).
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let dead_color = vec3<f32>(0.01, 0.012, 0.02);
    let tile_color = vec3<f32>(0.03, 0.04, 0.09);
    let edge_color = vec3<f32>(0.06, 0.08, 0.18);
    let live_color = vec3<f32>(0.35, 0.95, 0.55);

    let world = vec2<i32>(floor(frag.xy)) + view.screen_origin;
    let tile = world >> vec2<u32>(TILE_SHIFT);                // floor division by TILE
    let local = vec2<u32>(world & vec2<i32>(i32(TILE) - 1));  // always 0..TILE-1
    let m = tile - view.map_origin;
    if (m.x < 0 || m.y < 0 || u32(m.x) >= view.map_size.x || u32(m.y) >= view.map_size.y) {
        return vec4<f32>(dead_color, 1.0);
    }
    let slot = tile_map[u32(m.y) * view.map_size.x + u32(m.x)];
    if (slot == EMPTY) {
        return vec4<f32>(dead_color, 1.0);
    }
    let word = cells[slot * TILE_WORDS + local.y * WORDS_PER_ROW + (local.x >> 5u)];
    if (((word >> (local.x & 31u)) & 1u) == 1u) {
        return vec4<f32>(live_color, 1.0);
    }
    if (view.show_tiles == 1u) {
        let edge = local.x == 0u || local.y == 0u;
        return vec4<f32>(select(tile_color, edge_color, edge), 1.0);
    }
    return vec4<f32>(dead_color, 1.0);
}
