// Text overlay drawn with a tiny 3x5 bitmap font.
//
// The CPU lays text out on a `cols` x `rows` character grid, converts each
// character to a 15-bit glyph mask (bit = row * 3 + col, row 0 at the top) and
// uploads them in `glyphs`. This shader draws one quad covering the text box and
// decides per pixel whether it's text or background.

struct Overlay {
    screen: vec2<f32>,   // surface size in pixels
    origin: vec2<f32>,   // top-left of the box in pixels
    scale: f32,          // screen pixels per font pixel
    cols: u32,           // characters per line
    rows: u32,           // number of lines
    _pad: u32,
    glyphs: array<vec4<u32>, 64>, // up to 256 glyph masks, 4 per vec4, row-major
}

@group(0) @binding(0) var<uniform> ov: Overlay;

const GLYPH_W: f32 = 3.0;
const GLYPH_H: f32 = 5.0;
const CELL_W: f32 = 4.0; // glyph + 1 px spacing
const CELL_H: f32 = 7.0; // glyph + 2 px line spacing
const PAD: f32 = 2.0;    // padding around the text, in font pixels

fn box_size() -> vec2<f32> {
    let text = vec2<f32>(f32(ov.cols) * CELL_W - 1.0, f32(ov.rows) * CELL_H - 2.0);
    return (text + 2.0 * PAD) * ov.scale;
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Two triangles: (0,0) (1,0) (0,1) / (0,1) (1,0) (1,1)
    var corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    );
    let size = select(box_size(), vec2<f32>(0.0), ov.rows == 0u);
    let px = ov.origin + corners[i] * size;
    let ndc = px / ov.screen * 2.0 - 1.0;
    return vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let background = vec4<f32>(0.0, 0.0, 0.0, 0.7);
    let text = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    // Position in font pixels, relative to the first glyph's top-left.
    let p = floor((frag.xy - ov.origin) / ov.scale) - vec2<f32>(PAD);
    if (p.x < 0.0 || p.y < 0.0) {
        return background;
    }
    let col = u32(p.x / CELL_W);
    let row = u32(p.y / CELL_H);
    let gx = u32(p.x - f32(col) * CELL_W);
    let gy = u32(p.y - f32(row) * CELL_H);
    if (col >= ov.cols || row >= ov.rows || f32(gx) >= GLYPH_W || f32(gy) >= GLYPH_H) {
        return background;
    }
    let ci = row * ov.cols + col;
    let mask = ov.glyphs[ci / 4u][ci % 4u];
    let bit = gy * 3u + gx;
    return select(background, text, ((mask >> bit) & 1u) == 1u);
}
