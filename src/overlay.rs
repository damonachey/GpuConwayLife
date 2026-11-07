//! On-screen text overlay, drawn by `overlay.wgsl` with a built-in 3x5 pixel font.

const MAX_CHARS: usize = 256;
/// Screen pixels per font pixel, before HiDPI scaling.
const BASE_SCALE: f32 = 3.0;
/// Distance of the box from the window's top-left corner, in logical pixels.
const MARGIN: f32 = 8.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    screen: [f32; 2],
    origin: [f32; 2],
    scale: f32,
    cols: u32,
    rows: u32,
    _pad: u32,
    glyphs: [[u32; 4]; MAX_CHARS / 4],
}

pub struct Overlay {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
    uniforms: Uniforms,
    text: String,
    dirty: bool,
}

impl Overlay {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("overlay.wgsl"));
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay uniforms"),
            size: size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("overlay pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                ..Default::default()
            })),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            buffer,
            uniforms: bytemuck::Zeroable::zeroed(),
            text: String::new(),
            dirty: true,
        }
    }

    /// Call when the surface size or DPI scale changes.
    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f64) {
        let s = scale_factor as f32;
        self.uniforms.screen = [width as f32, height as f32];
        self.uniforms.origin = [MARGIN * s, MARGIN * s];
        self.uniforms.scale = (BASE_SCALE * s).round().max(1.0);
        self.dirty = true;
    }

    /// Set the overlay text. Lines are separated by `\n`; unchanged text is a no-op.
    /// Text beyond the 256-character grid is cut off.
    pub fn set_text(&mut self, text: &str) {
        if text == self.text {
            return;
        }
        self.text = text.to_owned();

        let lines: Vec<&str> = text.lines().collect();
        let cols = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0).max(1);
        let rows = lines.len().min(MAX_CHARS / cols);

        self.uniforms.glyphs = [[0; 4]; MAX_CHARS / 4];
        for (r, line) in lines.iter().take(rows).enumerate() {
            for (c, ch) in line.chars().take(cols).enumerate() {
                let i = r * cols + c;
                self.uniforms.glyphs[i / 4][i % 4] = glyph(ch);
            }
        }
        self.uniforms.cols = cols as u32;
        self.uniforms.rows = rows as u32;
        self.dirty = true;
    }

    /// Upload any changes and draw into `pass`.
    pub fn draw(&mut self, queue: &wgpu::Queue, pass: &mut wgpu::RenderPass<'_>) {
        if self.dirty {
            queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&self.uniforms));
            self.dirty = false;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..6, 0..1);
    }
}

/// 3x5 glyph mask for `ch`: bit `row * 3 + col`, row 0 at the top. Unknown → blank.
fn glyph(ch: char) -> u32 {
    let rows: [&str; 5] = match ch.to_ascii_uppercase() {
        '0' => ["###", "#.#", "#.#", "#.#", "###"],
        '1' => [".#.", "##.", ".#.", ".#.", "###"],
        '2' => ["###", "..#", "###", "#..", "###"],
        '3' => ["###", "..#", "###", "..#", "###"],
        '4' => ["#.#", "#.#", "###", "..#", "..#"],
        '5' => ["###", "#..", "###", "..#", "###"],
        '6' => ["###", "#..", "###", "#.#", "###"],
        '7' => ["###", "..#", "..#", "..#", "..#"],
        '8' => ["###", "#.#", "###", "#.#", "###"],
        '9' => ["###", "#.#", "###", "..#", "###"],
        'A' => [".#.", "#.#", "###", "#.#", "#.#"],
        'B' => ["##.", "#.#", "##.", "#.#", "##."],
        'C' => [".##", "#..", "#..", "#..", ".##"],
        'D' => ["##.", "#.#", "#.#", "#.#", "##."],
        'E' => ["###", "#..", "##.", "#..", "###"],
        'F' => ["###", "#..", "##.", "#..", "#.."],
        'G' => [".##", "#..", "#.#", "#.#", ".##"],
        'H' => ["#.#", "#.#", "###", "#.#", "#.#"],
        'I' => ["###", ".#.", ".#.", ".#.", "###"],
        'J' => ["..#", "..#", "..#", "#.#", ".#."],
        'K' => ["#.#", "#.#", "##.", "#.#", "#.#"],
        'L' => ["#..", "#..", "#..", "#..", "###"],
        'M' => ["#.#", "###", "###", "#.#", "#.#"],
        'N' => ["##.", "#.#", "#.#", "#.#", "#.#"],
        'O' => [".#.", "#.#", "#.#", "#.#", ".#."],
        'P' => ["##.", "#.#", "##.", "#..", "#.."],
        'Q' => [".#.", "#.#", "#.#", "##.", ".##"],
        'R' => ["##.", "#.#", "##.", "#.#", "#.#"],
        'S' => [".##", "#..", ".#.", "..#", "##."],
        'T' => ["###", ".#.", ".#.", ".#.", ".#."],
        'U' => ["#.#", "#.#", "#.#", "#.#", "###"],
        'V' => ["#.#", "#.#", "#.#", "#.#", ".#."],
        'W' => ["#.#", "#.#", "###", "###", "#.#"],
        'X' => ["#.#", "#.#", ".#.", "#.#", "#.#"],
        'Y' => ["#.#", "#.#", ".#.", ".#.", ".#."],
        'Z' => ["###", "..#", ".#.", "#..", "###"],
        '.' => ["...", "...", "...", "...", ".#."],
        ',' => ["...", "...", "...", ".#.", "#.."],
        ':' => ["...", ".#.", "...", ".#.", "..."],
        '-' => ["...", "...", "###", "...", "..."],
        '+' => ["...", ".#.", "###", ".#.", "..."],
        '/' => ["..#", "..#", ".#.", "#..", "#.."],
        '=' => ["...", "###", "...", "###", "..."],
        _ => return 0,
    };
    let mut mask = 0;
    for (r, row) in rows.iter().enumerate() {
        for (c, b) in row.bytes().enumerate() {
            if b == b'#' {
                mask |= 1 << (r * 3 + c);
            }
        }
    }
    mask
}
