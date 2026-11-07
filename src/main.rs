//! GPU Conway's Game of Life.
//!
//! The simulation runs in a wgpu compute shader (`life.wgsl`) and is drawn by a
//! fragment shader (`render.wgsl`) that reads the same GPU buffer, one cell per
//! screen pixel. The CPU only seeds generations; after that it just submits commands.
//!
//! Controls: Space pause/resume, N step one generation (while paused),
//! R reseed, 1 glider gun, 2 acorn, 3 bunnies, 4 Ark1, +/- change speed.

mod overlay;
mod patterns;

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const GRID_WIDTH: u32 = 1000;
const GRID_HEIGHT: u32 = 1000;
const WORKGROUP_SIZE: u32 = 8;

/// Speed is 2^level generations per frame, so -4 is one generation every 16 frames.
const MIN_SPEED_LEVEL: i32 = -4;
const MAX_SPEED_LEVEL: i32 = 10;
/// Largest number of generations recorded into one queue submission.
const MAX_GENERATIONS_PER_SUBMIT: u32 = 256;
/// How often FPS and generations/second are recalculated.
const STATS_PERIOD: Duration = Duration::from_millis(500);

/// Grid size, shared by the compute shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
}

/// Where the grid sits on screen, for the render shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct View {
    grid: [u32; 2],
    offset: [i32; 2],
}

/// Simulation state controlled from the keyboard.
struct Sim {
    paused: bool,
    speed_level: i32,
    /// Fractional generations owed, for speeds below one per frame.
    pending: f64,
    /// Generations requested with N while paused.
    single_steps: u32,
    generation: u64,
}

impl Sim {
    fn gens_per_frame(&self) -> f64 {
        2f64.powi(self.speed_level)
    }

    fn speed_label(&self) -> String {
        if self.speed_level >= 0 {
            format!("{}", 1u32 << self.speed_level)
        } else {
            format!("1/{}", 1u32 << -self.speed_level)
        }
    }

    /// How many generations to run this frame.
    fn take_generations(&mut self) -> u32 {
        if self.paused {
            return std::mem::take(&mut self.single_steps);
        }
        self.pending += self.gens_per_frame();
        let n = self.pending.floor();
        self.pending -= n;
        n as u32
    }
}

/// Rolling FPS and generations/second.
struct Stats {
    sample_start: Instant,
    frames: u32,
    generations: u64,
    fps: f64,
    gens_per_sec: f64,
}

impl Stats {
    fn record(&mut self, generations: u32) {
        self.frames += 1;
        self.generations += u64::from(generations);
        let elapsed = self.sample_start.elapsed();
        if elapsed >= STATS_PERIOD {
            let secs = elapsed.as_secs_f64();
            self.fps = f64::from(self.frames) / secs;
            self.gens_per_sec = self.generations as f64 / secs;
            self.frames = 0;
            self.generations = 0;
            self.sample_start = Instant::now();
        }
    }
}

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    cell_buffers: [wgpu::Buffer; 2],
    view_buffer: wgpu::Buffer,
    /// `compute_bind_groups[i]` reads buffer `i` and writes buffer `1 - i`.
    compute_bind_groups: [wgpu::BindGroup; 2],
    /// `render_bind_groups[i]` draws buffer `i`.
    render_bind_groups: [wgpu::BindGroup; 2],
    /// Index of the buffer that holds the current generation.
    current: usize,
    overlay: overlay::Overlay,
    sim: Sim,
    stats: Stats,
}

impl Gpu {
    async fn new(window: Arc<Window>) -> Self {
        // Honour wgpu's standard env vars, e.g. WGPU_BACKEND=dx12 or vulkan.
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("no suitable GPU adapter");
        log::info!("Using adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("request device");

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by adapter");
        surface.configure(&device, &config);

        // --- Buffers -------------------------------------------------------

        let params = Params { width: GRID_WIDTH, height: GRID_HEIGHT };
        let params_buffer = create_buffer_init(
            &device,
            "params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        let view_buffer = create_buffer_init(
            &device,
            "view",
            bytemuck::bytes_of(&view_for(config.width, config.height)),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let seed = random_cells(GRID_WIDTH * GRID_HEIGHT, time_seed());
        let cell_buffers = [0, 1].map(|i| {
            create_buffer_init(
                &device,
                &format!("cells {i}"),
                bytemuck::cast_slice(&seed),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )
        });

        // --- Compute pipeline ----------------------------------------------

        let life_shader = device.create_shader_module(wgpu::include_wgsl!("life.wgsl"));
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute layout"),
            entries: &[
                layout_entry(0, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Uniform),
                layout_entry(
                    1,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                layout_entry(
                    2,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
            ],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("life step"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute pipeline layout"),
                bind_group_layouts: &[Some(&compute_layout)],
                ..Default::default()
            })),
            module: &life_shader,
            entry_point: Some("step"),
            compilation_options: Default::default(),
            cache: None,
        });
        let compute_bind_groups = [0, 1].map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("compute {i} -> {}", 1 - i)),
                layout: &compute_layout,
                entries: &[
                    bind(0, &params_buffer),
                    bind(1, &cell_buffers[i]),
                    bind(2, &cell_buffers[1 - i]),
                ],
            })
        });

        // --- Render pipeline -----------------------------------------------

        let render_shader = device.create_shader_module(wgpu::include_wgsl!("render.wgsl"));
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render layout"),
            entries: &[
                layout_entry(0, wgpu::ShaderStages::FRAGMENT, wgpu::BufferBindingType::Uniform),
                layout_entry(
                    1,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
            ],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("draw cells"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render pipeline layout"),
                bind_group_layouts: &[Some(&render_layout)],
                ..Default::default()
            })),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let render_bind_groups = [0, 1].map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("render {i}")),
                layout: &render_layout,
                entries: &[bind(0, &view_buffer), bind(1, &cell_buffers[i])],
            })
        });

        let mut overlay = overlay::Overlay::new(&device, config.format);
        overlay.resize(config.width, config.height, window.scale_factor());

        Self {
            window,
            surface,
            device,
            queue,
            config,
            compute_pipeline,
            render_pipeline,
            cell_buffers,
            view_buffer,
            compute_bind_groups,
            render_bind_groups,
            current: 0,
            overlay,
            sim: Sim { paused: false, speed_level: 0, pending: 0.0, single_steps: 0, generation: 0 },
            stats: Stats {
                sample_start: Instant::now(),
                frames: 0,
                generations: 0,
                fps: 0.0,
                gens_per_sec: 0.0,
            },
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.overlay.resize(width, height, self.window.scale_factor());
            self.queue.write_buffer(
                &self.view_buffer,
                0,
                bytemuck::bytes_of(&view_for(width, height)),
            );
        }
    }

    fn key(&mut self, event: &KeyEvent) {
        log::debug!("key {:?} {:?} repeat={}", event.logical_key, event.state, event.repeat);
        if event.state != ElementState::Pressed {
            return;
        }
        let sim = &mut self.sim;
        match &event.logical_key {
            Key::Named(NamedKey::Space) if !event.repeat => {
                sim.paused = !sim.paused;
                sim.pending = 0.0;
            }
            // N may be held down to step repeatedly.
            Key::Character(c) if c.eq_ignore_ascii_case("n") && sim.paused => {
                sim.single_steps += 1;
            }
            Key::Character(c) if c.eq_ignore_ascii_case("r") && !event.repeat => self.reseed(),
            Key::Character(c) if c == "1" && !event.repeat => {
                self.load_pattern(patterns::GOSPER_GLIDER_GUN);
            }
            Key::Character(c) if c == "2" && !event.repeat => self.load_pattern(patterns::ACORN),
            Key::Character(c) if c == "3" && !event.repeat => self.load_pattern(patterns::BUNNIES),
            Key::Character(c) if c == "4" && !event.repeat => self.load_pattern(patterns::ARK1),
            Key::Character(c) if c == "+" || c == "=" => {
                sim.speed_level = (sim.speed_level + 1).min(MAX_SPEED_LEVEL);
            }
            Key::Character(c) if c == "-" || c == "_" => {
                sim.speed_level = (sim.speed_level - 1).max(MIN_SPEED_LEVEL);
            }
            _ => {}
        }
    }

    /// Clear the grid and place `pattern` in the centre.
    fn load_pattern(&mut self, pattern: &[&str]) {
        self.load_cells(&patterns::centered(pattern, GRID_WIDTH, GRID_HEIGHT));
    }

    /// Replace the current generation with fresh random cells.
    fn reseed(&mut self) {
        self.load_cells(&random_cells(GRID_WIDTH * GRID_HEIGHT, time_seed()));
    }

    /// Replace the current generation with `cells` and restart the generation count.
    fn load_cells(&mut self, cells: &[u32]) {
        self.queue.write_buffer(&self.cell_buffers[self.current], 0, bytemuck::cast_slice(cells));
        self.sim.generation = 0;
        self.sim.pending = 0.0;
    }

    fn overlay_text(&self) -> String {
        let (sim, stats) = (&self.sim, &self.stats);
        let ms = if stats.fps > 0.0 { 1000.0 / stats.fps } else { 0.0 };
        let state = if sim.paused { "  PAUSED" } else { "" };
        format!(
            "{:.0} FPS  {ms:.2} MS\n\
             GEN {}  {:.0} GEN/S\n\
             SPEED {} GEN/FRAME{state}\n\
             GRID {GRID_WIDTH}X{GRID_HEIGHT}\n\
             SPACE PAUSE  N STEP  R RESEED  +/- SPEED\n\
             1 GUN  2 ACORN  3 BUNNIES  4 ARK1",
            stats.fps,
            sim.generation,
            stats.gens_per_sec,
            sim.speed_label(),
        )
    }

    /// Advance the simulation as needed and draw a frame.
    fn frame(&mut self) {
        use wgpu::CurrentSurfaceTexture as Cst;
        let frame = match self.surface.get_current_texture() {
            Cst::Success(frame) => frame,
            Cst::Suboptimal(frame) => {
                // Draw this one, but reconfigure so the next frame is a perfect fit.
                self.surface.configure(&self.device, &self.config);
                frame
            }
            Cst::Outdated | Cst::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Cst::Timeout | Cst::Occluded => return,
            Cst::Validation => {
                log::warn!("surface validation error; skipping frame");
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());

        let generations = self.sim.take_generations();
        // Submit the simulation in chunks: the NVIDIA Vulkan driver (560.94) crashes
        // with STATUS_INTEGER_DIVIDE_BY_ZERO when one submission holds ~1024 dispatches.
        let mut remaining = generations;
        while remaining > 0 {
            let chunk = remaining.min(MAX_GENERATIONS_PER_SUBMIT);
            remaining -= chunk;
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                // Each dispatch is its own synchronization scope, so ping-ponging
                // between bind groups inside one pass is safe.
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.compute_pipeline);
                for _ in 0..chunk {
                    pass.set_bind_group(0, &self.compute_bind_groups[self.current], &[]);
                    pass.dispatch_workgroups(
                        GRID_WIDTH.div_ceil(WORKGROUP_SIZE),
                        GRID_HEIGHT.div_ceil(WORKGROUP_SIZE),
                        1,
                    );
                    self.current = 1 - self.current;
                }
            }
            self.queue.submit([encoder.finish()]);
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.sim.generation += u64::from(generations);
        self.stats.record(generations);
        self.overlay.set_text(&self.overlay_text());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.render_bind_groups[self.current], &[]);
            pass.draw(0..3, 0..1);
            self.overlay.draw(&self.queue, &mut pass);
        }

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.queue.present(frame);
    }
}

/// Centre the grid in a `width` x `height` surface.
fn view_for(width: u32, height: u32) -> View {
    View {
        grid: [GRID_WIDTH, GRID_HEIGHT],
        offset: [
            (width as i32 - GRID_WIDTH as i32) / 2,
            (height as i32 - GRID_HEIGHT as i32) / 2,
        ],
    }
}

fn create_buffer_init(
    device: &wgpu::Device,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage,
    })
}

fn layout_entry(
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

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
}

fn time_seed() -> u32 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    nanos | 1 // xorshift state must be non-zero
}

/// Seed roughly 25% of cells alive using a small xorshift generator.
fn random_cells(count: u32, mut state: u32) -> Vec<u32> {
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            u32::from(state % 4 == 0)
        })
        .collect()
}

#[derive(Default)]
struct App {
    gpu: Option<Gpu>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        // Physical size so one cell maps to exactly one screen pixel.
        let attrs = Window::default_attributes()
            .with_title("GPU Conway's Game of Life")
            .with_inner_size(winit::dpi::PhysicalSize::new(GRID_WIDTH, GRID_HEIGHT));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.gpu = Some(pollster::block_on(Gpu::new(window)));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => gpu.resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => gpu.key(&event),
            WindowEvent::RedrawRequested => gpu.frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gpu) = &self.gpu {
            gpu.window.request_redraw();
        }
    }
}

fn main() {
    // Vulkan loader chatter (e.g. stale overlay layers in the registry) is muted by
    // default; set RUST_LOG to override.
    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("warn,gpu_conway_life=info,wgpu_hal::vulkan::instance=off"),
    )
    .init();
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default()).expect("run event loop");
}
