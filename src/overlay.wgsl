// Text overlay drawn with a tiny 3x5 bitmap font.
//
// The CPU converts each character to a 15-bit glyph mask (bit = row * 3 + col,
// row 0 at the top) and uploads them in `glyphs`. This shader draws one quad
// covering the text box and decides per pixel whether it's text or background.

struct Overlay {
    screen: vec2<f32>,   // surface size in pixels
    origin: vec2<f32>,   // top-left of the box in pixels
    scale: f32,          // screen pixels per font pixel
    count: u32,          // number of characters
    glyphs: array<vec4<u32>, 8>, // up to 32 glyph masks, 4 per vec4
}

@group(0) @binding(0) var<uniform> ov: Overlay;

const CELL_W: f32 = 4.0; // 3 px glyph + 1 px spacing
const GLYPH_H: f32 = 5.0;
const PAD: f32 = 1.0;    // padding around the text, in font pixels

fn box_size() -> vec2<f32> {
    return vec2<f32>(f32(ov.count) * CELL_W - 1.0 + 2.0 * PAD, GLYPH_H + 2.0 * PAD) * ov.scale;
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Two triangles: (0,0) (1,0) (0,1) / (0,1) (1,0) (1,1)
    var corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    );
    let px = ov.origin + corners[i] * box_size();
    let ndc = px / ov.screen * 2.0 - 1.0;
    return vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let background = vec4<f32>(0.0, 0.0, 0.0, 0.65);
    let text = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    // Position in font pixels, relative to the first glyph's top-left.
    let p = floor((frag.xy - ov.origin) / ov.scale) - vec2<f32>(PAD);
    if (p.x < 0.0 || p.y < 0.0 || p.y >= GLYPH_H) {
        return background;
    }
    let ci = u32(p.x / CELL_W);
    let gx = u32(p.x) % 4u;
    if (ci >= ov.count || gx >= 3u) {
        return background;
    }
    let mask = ov.glyphs[ci / 4u][ci % 4u];
    let bit = u32(p.y) * 3u + gx;
    return select(background, text, ((mask >> bit) & 1u) == 1u);
}
