//! GPU Conway's Game of Life on a huge, bit-packed grid.
//!
//! Cells are stored one bit each, 32 to a `u32`. The grid is as large as the GPU
//! allows (each of the two ping-pong buffers up to `MAX_BUFFER_BYTES`), or the size
//! given by the `LIFE_GRID` environment variable. The simulation runs in a wgpu
//! compute shader (`life.wgsl`) and is drawn by a fragment shader (`render.wgsl`)
//! that reads the same GPU buffer, one cell per screen pixel, centred on the grid's
//! middle. Random soup is generated on the GPU too.
//!
//! Controls: Space pause/resume, N step one generation (while paused),
//! R reseed, 1 glider gun, 2 acorn, 3 bunnies, 4 Ark1, +/- change speed.

mod overlay;
mod patterns;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Upper bound on each of the two cell buffers (most GPUs allow ~2 GiB per binding).
const MAX_BUFFER_BYTES: u64 = 2 << 30;
/// Default grid side: 32768² ≈ 1.07 billion cells (128 MiB per buffer). Big enough
/// for most patterns to run free, small enough to stay fast. `LIFE_GRID` can go
/// higher, up to what the GPU allows.
const DEFAULT_GRID_SIDE: u32 = 32768;
/// Matches `@workgroup_size(64, 4)` in life.wgsl: 64 words across, 4 rows down.
const WORKGROUP_SIZE: (u32, u32) = (64, 4);

/// Speed is 2^level generations per frame, so -4 is one generation every 16 frames.
const MIN_SPEED_LEVEL: i32 = -4;
const MAX_SPEED_LEVEL: i32 = 10;
/// Largest number of generations recorded into one queue submission. (The NVIDIA
/// 560.94 Vulkan driver crashed with ~1024 dispatches in one submission.)
const MAX_GENERATIONS_PER_SUBMIT: u32 = 64;
/// Cap on simulation work per frame, so huge grids at high speed stay responsive.
/// Generations per frame are limited to what the GPU did in this much time recently.
const FRAME_BUDGET: Duration = Duration::from_millis(100);
/// How often FPS and generations/second are recalculated.
const STATS_PERIOD: Duration = Duration::from_millis(500);

/// Grid dimensions. Width is a multiple of 32 so rows are whole words.
#[derive(Clone, Copy, Debug)]
struct Grid {
    width: u32,
    height: u32,
}

impl Grid {
    /// Largest square grid whose bit-packed buffer fits in `bytes`.
    fn square_fitting(bytes: u64) -> Self {
        let side = ((bytes * 8) as f64).sqrt() as u32;
        Self::square(side)
    }

    /// A square grid of about `side` cells (rounded down to a multiple of 32).
    fn square(side: u32) -> Self {
        let side = (side / 32).max(1) * 32;
        Self { width: side, height: side }
    }

    fn words_per_row(&self) -> u32 {
        self.width / 32
    }

    fn bytes(&self) -> u64 {
        u64::from(self.words_per_row()) * 4 * u64::from(self.height)
    }

    fn cells(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    fn dispatch_size(&self) -> (u32, u32) {
        (self.words_per_row().div_ceil(WORKGROUP_SIZE.0), self.height.div_ceil(WORKGROUP_SIZE.1))
    }
}

/// Compute shader parameters; matches `Params` in life.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    words_per_row: u32,
    seed: u32,
}

/// Where the grid sits on screen, for the render shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct View {
    grid: [u32; 2],
    offset: [i32; 2],
    words_per_row: u32,
    _pad: u32,
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

/// Matches COUNTERS in life.wgsl.
const POP_COUNTERS: u64 = 1024;
/// How often the population is measured.
const POPULATION_PERIOD: Duration = Duration::from_millis(500);

/// Background population counting: a GPU pass writes partial counts, which are
/// copied to a staging buffer and read back without stalling the frame loop.
struct Population {
    /// Generation being counted, while a readback is in flight.
    in_flight: Option<u64>,
    /// Set by the map callback once the staging buffer is readable.
    mapped: Arc<std::sync::atomic::AtomicBool>,
    last_request: Instant,
    /// Latest result: (live cells, generation).
    value: Option<(u64, u64)>,
}

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    compute_pipeline: wgpu::ComputePipeline,
    seed_pipeline: wgpu::ComputePipeline,
    count_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    grid: Grid,
    params_buffer: wgpu::Buffer,
    count_buffer: wgpu::Buffer,
    count_staging: wgpu::Buffer,
    population: Population,
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

        // Ask for the biggest buffers the GPU supports (wgpu defaults to 128 MiB).
        let adapter_limits = adapter.limits();
        let max_buffer = MAX_BUFFER_BYTES
            .min(adapter_limits.max_storage_buffer_binding_size)
            .min(adapter_limits.max_buffer_size);
        let defaults = wgpu::Limits::default();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: max_buffer
                .max(defaults.max_storage_buffer_binding_size),
            max_buffer_size: max_buffer.max(defaults.max_buffer_size),
            ..defaults
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("life device"),
                required_limits,
                ..Default::default()
            })
            .await
            .expect("request device");

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by adapter");
        surface.configure(&device, &config);

        // --- Buffers -------------------------------------------------------

        // Grid size: LIFE_GRID=<side> if set, else DEFAULT_GRID_SIDE; either way no
        // bigger than the GPU's limits allow.
        let requested = std::env::var("LIFE_GRID").ok().and_then(|s| s.trim().parse::<u32>().ok());
        let mut grid = Grid::square(requested.unwrap_or(DEFAULT_GRID_SIDE));
        if grid.bytes() > max_buffer {
            log::warn!("{0}x{0} grid is too big for this GPU; shrinking", grid.width);
            grid = Grid::square_fitting(max_buffer);
        }
        // The limits say what's addressable, not what's free: if allocation fails,
        // shrink the grid and try again.
        let cell_buffers = loop {
            let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let buffers = [0, 1].map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("cells {i}")),
                    size: grid.bytes(),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
            match scope.pop().await {
                None => break buffers,
                Some(e) if grid.width > 1024 => {
                    log::warn!("{}x{} grid didn't fit ({e}); shrinking", grid.width, grid.height);
                    grid = Grid::square(grid.width * 3 / 4);
                }
                Some(e) => panic!("can't allocate even a small grid: {e}"),
            }
        };
        log::info!(
            "Grid {}x{} ({:.1} billion cells), {} MiB per buffer x 2",
            grid.width,
            grid.height,
            grid.cells() as f64 / 1e9,
            grid.bytes() >> 20
        );

        let params = Params {
            width: grid.width,
            height: grid.height,
            words_per_row: grid.words_per_row(),
            seed: 0,
        };
        let params_buffer = create_buffer_init(
            &device,
            "params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let view_buffer = create_buffer_init(
            &device,
            "view",
            bytemuck::bytes_of(&view_for(config.width, config.height, grid)),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("population counters"),
            size: POP_COUNTERS * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("population readback"),
            size: POP_COUNTERS * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
                layout_entry(
                    3,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
            ],
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute pipeline layout"),
                bind_group_layouts: &[Some(&compute_layout)],
                ..Default::default()
            });
        let make_compute = |label: &str, entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&compute_pipeline_layout),
                module: &life_shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let compute_pipeline = make_compute("life step", "step");
        let seed_pipeline = make_compute("random seed", "seed");
        let count_pipeline = make_compute("population count", "count");
        let compute_bind_groups = [0, 1].map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("compute {i} -> {}", 1 - i)),
                layout: &compute_layout,
                entries: &[
                    bind(0, &params_buffer),
                    bind(1, &cell_buffers[i]),
                    bind(2, &cell_buffers[1 - i]),
                    bind(3, &count_buffer),
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

        let mut gpu = Self {
            window,
            surface,
            device,
            queue,
            config,
            compute_pipeline,
            seed_pipeline,
            count_pipeline,
            render_pipeline,
            grid,
            params_buffer,
            count_buffer,
            count_staging,
            population: Population {
                in_flight: None,
                mapped: Default::default(),
                last_request: Instant::now(),
                value: None,
            },
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
        };
        gpu.reseed();
        gpu
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
                bytemuck::bytes_of(&view_for(width, height, self.grid)),
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
        // Clear on the GPU and submit *before* writing the pattern: queue writes are
        // applied before the next submission, so they must come after this one.
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.cell_buffers[self.current], 0, None);
        self.queue.submit([encoder.finish()]);

        let (w, h) = (self.grid.width as i32, self.grid.height as i32);
        let mut words: BTreeMap<u32, u32> = BTreeMap::new();
        for (dx, dy) in patterns::centered(pattern) {
            let x = (w / 2 + dx).rem_euclid(w) as u32;
            let y = (h / 2 + dy).rem_euclid(h) as u32;
            *words.entry(y * self.grid.words_per_row() + x / 32).or_default() |= 1 << (x % 32);
        }
        for (index, word) in words {
            self.queue.write_buffer(
                &self.cell_buffers[self.current],
                u64::from(index) * 4,
                bytemuck::bytes_of(&word),
            );
        }
        self.restart();
    }

    /// Fill the current generation with random soup, generated on the GPU.
    fn reseed(&mut self) {
        let params = Params {
            width: self.grid.width,
            height: self.grid.height,
            words_per_row: self.grid.words_per_row(),
            seed: time_seed(),
        };
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.seed_pipeline);
            // This bind group writes `dst` = the current buffer.
            pass.set_bind_group(0, &self.compute_bind_groups[1 - self.current], &[]);
            let (x, y) = self.grid.dispatch_size();
            pass.dispatch_workgroups(x, y, 1);
        }
        self.queue.submit([encoder.finish()]);
        self.restart();
    }

    fn restart(&mut self) {
        self.sim.generation = 0;
        self.sim.pending = 0.0;
        // Count the new world promptly (an in-flight count of the old one is still
        // collected, but labelled with its own generation).
        self.population.value = None;
        self.population.last_request = Instant::now() - POPULATION_PERIOD;
    }

    /// Collect a finished population readback, and start a new one when due.
    /// Must be called after this frame's generations have been submitted.
    fn update_population(&mut self) {
        use std::sync::atomic::Ordering;
        let _ = self.device.poll(wgpu::PollType::Poll);
        let pop = &mut self.population;
        if let Some(generation) = pop.in_flight {
            if !pop.mapped.load(Ordering::Acquire) {
                return;
            }
            let total: u64 = {
                let view = self.count_staging.get_mapped_range(..).expect("mapped counts");
                bytemuck::cast_slice::<u8, u32>(&view).iter().map(|&c| u64::from(c)).sum()
            };
            self.count_staging.unmap();
            pop.in_flight = None;
            pop.value = Some((total, generation));
        }
        if pop.last_request.elapsed() < POPULATION_PERIOD {
            return;
        }
        pop.last_request = Instant::now();
        pop.in_flight = Some(self.sim.generation);
        pop.mapped.store(false, Ordering::Release);

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.count_buffer, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.count_pipeline);
            // This bind group's `src` is the current generation.
            pass.set_bind_group(0, &self.compute_bind_groups[self.current], &[]);
            let (x, y) = self.grid.dispatch_size();
            pass.dispatch_workgroups(x, y, 1);
        }
        encoder.copy_buffer_to_buffer(&self.count_buffer, 0, &self.count_staging, 0, None);
        self.queue.submit([encoder.finish()]);
        let mapped = pop.mapped.clone();
        self.count_staging.map_async(wgpu::MapMode::Read, .., move |result| {
            if result.is_ok() {
                mapped.store(true, Ordering::Release);
            }
        });
    }

    fn overlay_text(&self) -> String {
        let (sim, stats) = (&self.sim, &self.stats);
        let ms = if stats.fps > 0.0 { 1000.0 / stats.fps } else { 0.0 };
        let state = if sim.paused { "  PAUSED" } else { "" };
        let cell_rate = human(stats.gens_per_sec * self.grid.cells() as f64, &[
            (1e12, "TRILLION"),
            (1e9, "BILLION"),
            (1e6, "MILLION"),
        ]);
        let memory = human(self.grid.bytes() as f64, &[
            (f64::from(1 << 30), "GB"),
            (f64::from(1 << 20), "MB"),
            (f64::from(1 << 10), "KB"),
        ]);
        let population = match self.population.value {
            Some((cells, generation)) => format!("POP {cells} AT GEN {generation}"),
            None => "POP COUNTING".to_owned(),
        };
        format!(
            "{:.0} FPS  {ms:.2} MS\n\
             GEN {}  {:.0} GEN/S  {cell_rate} CELLS/S\n\
             {population}\n\
             SPEED {} GEN/FRAME{state}\n\
             GRID {}X{}  {memory} X 2\n\
             SPACE PAUSE  N STEP  R RESEED  +/- SPEED\n\
             1 GUN  2 ACORN  3 BUNNIES  4 ARK1",
            stats.fps,
            sim.generation,
            stats.gens_per_sec,
            sim.speed_label(),
            self.grid.width,
            self.grid.height,
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

        let mut generations = self.sim.take_generations();
        if !self.sim.paused {
            // Don't queue more than ~FRAME_BUDGET of GPU work, judged by recent
            // throughput; if the GPU can't keep up with the requested speed, drop
            // the excess rather than falling further behind.
            let budget = (self.stats.gens_per_sec * FRAME_BUDGET.as_secs_f64()).max(1.0) as u32;
            generations = generations.min(budget.max(1));
        }
        let (dispatch_x, dispatch_y) = self.grid.dispatch_size();
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
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                    self.current = 1 - self.current;
                }
            }
            self.queue.submit([encoder.finish()]);
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.sim.generation += u64::from(generations);
        self.stats.record(generations);
        self.update_population();
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

/// Format `value` with the largest unit it reaches, e.g. "1.4 TRILLION" or "512.0 KB".
fn human(value: f64, units: &[(f64, &str)]) -> String {
    match units.iter().find(|(scale, _)| value >= *scale) {
        Some((scale, name)) => format!("{:.1} {name}", value / scale),
        None => format!("{value:.0}"),
    }
}

/// Draw the middle of `grid` at the middle of a `width` x `height` surface.
fn view_for(width: u32, height: u32, grid: Grid) -> View {
    View {
        grid: [grid.width, grid.height],
        offset: [
            (width as i32 - grid.width as i32) / 2,
            (height as i32 - grid.height as i32) / 2,
        ],
        words_per_row: grid.words_per_row(),
        _pad: 0,
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
    nanos
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
            .with_inner_size(winit::dpi::PhysicalSize::new(1000, 1000));
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
