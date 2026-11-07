//! The interface shared by the simulation engines, so the app can swap them.

/// GPU handles every engine needs. `Device` and `Queue` are cheap to clone.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Surface format the engines render into.
    pub format: wgpu::TextureFormat,
    /// Largest storage buffer the device was created to allow.
    pub max_buffer: u64,
}

/// A Game of Life simulation that lives on the GPU and can draw itself.
pub trait Engine {
    /// One overlay line naming the engine, e.g. "DENSE 16384X16384 ...".
    fn title(&self) -> String;
    /// Engine-specific overlay lines (memory, tiles, ...).
    fn info(&self) -> Vec<String>;
    fn generation(&self) -> u64;
    /// Latest population measurement: (live cells, generation it was taken at).
    fn population(&self) -> Option<(u64, u64)>;
    /// Cells the GPU processes per generation, for the cells/second readout.
    fn cells_per_generation(&self) -> u64;

    /// Replace the world with these live cells (offsets from the world's centre).
    fn load(&mut self, cells: &[(i32, i32)]);
    /// Replace the world with random soup.
    fn random(&mut self);
    /// Advance `n` generations.
    fn step(&mut self, n: u32);
    /// Once per frame, after stepping: refresh view state and background readbacks.
    fn update(&mut self, screen_width: u32, screen_height: u32);
    /// Draw the world into `pass` (which covers the whole surface).
    fn draw(&self, pass: &mut wgpu::RenderPass<'_>);
    /// Toggle the tile visualisation, for engines that have tiles.
    fn toggle_tiles(&mut self) {}
}

/// Conway's rule for 32 bit-packed cells, shared by both engines' compute shaders.
pub const LIFE_RULE_WGSL: &str = include_str!("life_rule.wgsl");

/// A shader module built from several WGSL sources joined in order (WGSL has no
/// `#include`), e.g. shared constants and functions followed by the shader itself.
pub fn shader(device: &wgpu::Device, label: &str, sources: &[&str]) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(sources.concat().into()),
    })
}

/// A buffer the CPU updates with `queue.write_buffer`, read as a uniform.
pub fn uniform_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    buffer(device, label, size, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST)
}

/// A storage buffer, plus any `extra` usages (copy source/destination, ...).
pub fn storage_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    buffer(device, label, size, wgpu::BufferUsages::STORAGE | extra)
}

/// A buffer the CPU maps to read GPU results back.
pub fn readback_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    buffer(device, label, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)
}

fn buffer(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// Shared helpers for building pipelines and bind groups.
pub fn layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

pub fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
}

/// A full-screen render pipeline from `module`'s `vs_main` / `fs_main`.
pub fn fullscreen_pipeline(
    ctx: &GpuContext,
    label: &str,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let device = &ctx.device;
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(layout)],
            ..Default::default()
        })),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(ctx.format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Format `value` with the largest unit it reaches, e.g. "1.4 TRILLION" or "512.0 KB".
pub fn human(value: f64, units: &[(f64, &str)]) -> String {
    match units.iter().find(|(scale, _)| value >= *scale) {
        Some((scale, name)) => format!("{:.1} {name}", value / scale),
        None => format!("{value:.0}"),
    }
}

pub const BYTE_UNITS: &[(f64, &str)] =
    &[(1073741824.0, "GB"), (1048576.0, "MB"), (1024.0, "KB")];

pub fn time_seed() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos()
        | 1
}
