//! Dense engine: one big bit-packed grid with dead edges.
//!
//! Cells are stored one bit each, 32 to a `u32`, in two ping-pong buffers. The
//! simulation runs in `dense_life.wgsl` (32 cells per thread via a bitwise adder)
//! and is drawn by `dense_render.wgsl`, one cell per screen pixel, centred on the
//! grid's middle. Random soup and population counts are computed on the GPU.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::engine::{BYTE_UNITS, Engine, GpuContext, bind, fullscreen_pipeline, human, layout_entry};

/// Default grid side: 16384² ≈ 268 million cells (32 MiB per buffer). Plenty of
/// room for most patterns' cores, and fast (~5,000 generations/s on an RTX 3070).
/// `LIFE_GRID` can go higher, up to what the GPU allows.
pub const DEFAULT_GRID_SIDE: u32 = 16384;
/// Matches `@workgroup_size(64, 4)` in dense_life.wgsl: 64 words across, 4 rows down.
const WORKGROUP_SIZE: (u32, u32) = (64, 4);
/// Largest number of generations recorded into one queue submission. (The NVIDIA
/// 560.94 Vulkan driver crashed with ~1024 dispatches in one submission.)
const MAX_GENERATIONS_PER_SUBMIT: u32 = 64;
/// Matches COUNTERS in dense_life.wgsl.
const POP_COUNTERS: u64 = 1024;
/// How often the population is measured.
const POPULATION_PERIOD: Duration = Duration::from_millis(500);

/// Grid dimensions. Width is a multiple of 32 so rows are whole words.
#[derive(Clone, Copy, Debug)]
struct Grid {
    width: u32,
    height: u32,
}

impl Grid {
    /// Largest square grid whose bit-packed buffer fits in `bytes`.
    fn square_fitting(bytes: u64) -> Self {
        Self::square(((bytes * 8) as f64).sqrt() as u32)
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

/// Compute shader parameters; matches `Params` in dense_life.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    words_per_row: u32,
    seed: u32,
}

/// Where the grid sits on screen; matches `View` in dense_render.wgsl.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct View {
    grid: [u32; 2],
    offset: [i32; 2],
    words_per_row: u32,
    _pad: u32,
}

/// Background population counting: a GPU pass writes partial counts, which are
/// copied to a staging buffer and read back without stalling the frame loop.
struct Population {
    /// Generation being counted, while a readback is in flight.
    in_flight: Option<u64>,
    /// Set by the map callback once the staging buffer is readable.
    mapped: Arc<AtomicBool>,
    last_request: Instant,
    /// Latest result: (live cells, generation).
    value: Option<(u64, u64)>,
}

pub struct DenseEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    grid: Grid,
    step_pipeline: wgpu::ComputePipeline,
    seed_pipeline: wgpu::ComputePipeline,
    count_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    params_buffer: wgpu::Buffer,
    cell_buffers: [wgpu::Buffer; 2],
    view_buffer: wgpu::Buffer,
    count_buffer: wgpu::Buffer,
    count_staging: wgpu::Buffer,
    /// `compute_bind_groups[i]` reads cell buffer `i` and writes buffer `1 - i`.
    compute_bind_groups: [wgpu::BindGroup; 2],
    /// `render_bind_groups[i]` draws cell buffer `i`.
    render_bind_groups: [wgpu::BindGroup; 2],
    /// Index of the cell buffer that holds the current generation.
    current: usize,
    generation: u64,
    population: Population,
    last_view: Option<View>,
}

impl DenseEngine {
    pub async fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;

        // Grid size: LIFE_GRID=<side> if set, else DEFAULT_GRID_SIDE; either way no
        // bigger than the GPU's limits allow.
        let requested = std::env::var("LIFE_GRID").ok().and_then(|s| s.trim().parse::<u32>().ok());
        let mut grid = Grid::square(requested.unwrap_or(DEFAULT_GRID_SIDE));
        if grid.bytes() > ctx.max_buffer {
            log::warn!("{0}x{0} grid is too big for this GPU; shrinking", grid.width);
            grid = Grid::square_fitting(ctx.max_buffer);
        }
        // The limits say what's addressable, not what's free: if allocation fails,
        // shrink the grid and try again.
        let cell_buffers = loop {
            let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let buffers = [0, 1].map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("dense cells {i}")),
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
            "Dense grid {}x{} ({:.2} billion cells), {} MiB per buffer x 2",
            grid.width,
            grid.height,
            grid.cells() as f64 / 1e9,
            grid.bytes() >> 20
        );

        let uniform = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let params_buffer = uniform("dense params", size_of::<Params>() as u64);
        let params = Params {
            width: grid.width,
            height: grid.height,
            words_per_row: grid.words_per_row(),
            seed: 0,
        };
        ctx.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
        let view_buffer = uniform("dense view", size_of::<View>() as u64);
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

        // --- Compute pipelines (step, seed, count share one layout) ---------

        use wgpu::BufferBindingType::{Storage, Uniform};
        let cs = wgpu::ShaderStages::COMPUTE;
        let life_shader = device.create_shader_module(wgpu::include_wgsl!("dense_life.wgsl"));
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dense compute layout"),
            entries: &[
                layout_entry(0, cs, Uniform),
                layout_entry(1, cs, Storage { read_only: true }),
                layout_entry(2, cs, Storage { read_only: false }),
                layout_entry(3, cs, Storage { read_only: false }),
            ],
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dense compute pipeline layout"),
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
        let step_pipeline = make_compute("dense step", "step");
        let seed_pipeline = make_compute("dense random seed", "seed");
        let count_pipeline = make_compute("dense population count", "count");
        let compute_bind_groups = [0, 1].map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("dense compute {i} -> {}", 1 - i)),
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

        let fs = wgpu::ShaderStages::FRAGMENT;
        let render_shader = device.create_shader_module(wgpu::include_wgsl!("dense_render.wgsl"));
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dense render layout"),
            entries: &[layout_entry(0, fs, Uniform), layout_entry(1, fs, Storage { read_only: true })],
        });
        let render_pipeline = fullscreen_pipeline(ctx, "dense draw", &render_shader, &render_layout);
        let render_bind_groups = [0, 1].map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("dense render {i}")),
                layout: &render_layout,
                entries: &[bind(0, &view_buffer), bind(1, &cell_buffers[i])],
            })
        });

        Self {
            device: ctx.device.clone(),
            queue: ctx.queue.clone(),
            grid,
            step_pipeline,
            seed_pipeline,
            count_pipeline,
            render_pipeline,
            params_buffer,
            cell_buffers,
            view_buffer,
            count_buffer,
            count_staging,
            compute_bind_groups,
            render_bind_groups,
            current: 0,
            generation: 0,
            population: Population {
                in_flight: None,
                mapped: Default::default(),
                last_request: Instant::now(),
                value: None,
            },
            last_view: None,
        }
    }

    fn restart(&mut self) {
        self.generation = 0;
        // Count the new world promptly (an in-flight count of the old one is still
        // collected, but labelled with its own generation).
        self.population.value = None;
        self.population.last_request = Instant::now() - POPULATION_PERIOD;
    }

    /// Collect a finished population readback, and start a new one when due.
    fn update_population(&mut self) {
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
        pop.in_flight = Some(self.generation);
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
}

impl Engine for DenseEngine {
    fn title(&self) -> String {
        "ENGINE DENSE: BIT-PACKED GRID, DEAD EDGES".to_owned()
    }

    fn info(&self) -> String {
        format!(
            "GRID {}X{}  {} X 2",
            self.grid.width,
            self.grid.height,
            human(self.grid.bytes() as f64, BYTE_UNITS)
        )
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn population(&self) -> Option<(u64, u64)> {
        self.population.value
    }

    fn cells_per_generation(&self) -> u64 {
        self.grid.cells()
    }

    fn load(&mut self, cells: &[(i32, i32)]) {
        // Clear on the GPU and submit *before* writing the pattern: queue writes are
        // applied before the next submission, so they must come after this one.
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.cell_buffers[self.current], 0, None);
        self.queue.submit([encoder.finish()]);

        let (w, h) = (self.grid.width as i32, self.grid.height as i32);
        let mut words: BTreeMap<u32, u32> = BTreeMap::new();
        for &(dx, dy) in cells {
            let (x, y) = (w / 2 + dx, h / 2 + dy);
            if !(0..w).contains(&x) || !(0..h).contains(&y) {
                continue; // off the grid: dead
            }
            let (x, y) = (x as u32, y as u32);
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

    fn random(&mut self) {
        let params = Params {
            width: self.grid.width,
            height: self.grid.height,
            words_per_row: self.grid.words_per_row(),
            seed: crate::engine::time_seed(),
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

    fn step(&mut self, n: u32) {
        let (dispatch_x, dispatch_y) = self.grid.dispatch_size();
        let mut remaining = n;
        while remaining > 0 {
            let chunk = remaining.min(MAX_GENERATIONS_PER_SUBMIT);
            remaining -= chunk;
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                // Each dispatch is its own synchronization scope, so ping-ponging
                // between bind groups inside one pass is safe.
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.step_pipeline);
                for _ in 0..chunk {
                    pass.set_bind_group(0, &self.compute_bind_groups[self.current], &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                    self.current = 1 - self.current;
                }
            }
            self.queue.submit([encoder.finish()]);
        }
        self.generation += u64::from(n);
    }

    fn update(&mut self, screen_width: u32, screen_height: u32) {
        // Draw the middle of the grid at the middle of the window.
        let view = View {
            grid: [self.grid.width, self.grid.height],
            offset: [
                (screen_width as i32 - self.grid.width as i32) / 2,
                (screen_height as i32 - self.grid.height as i32) / 2,
            ],
            words_per_row: self.grid.words_per_row(),
            _pad: 0,
        };
        if self.last_view != Some(view) {
            self.queue.write_buffer(&self.view_buffer, 0, bytemuck::bytes_of(&view));
            self.last_view = Some(view);
        }
        self.update_population();
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_bind_groups[self.current], &[]);
        pass.draw(0..3, 0..1);
    }
}
