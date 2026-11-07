//! Sparse engine: an effectively unbounded world made of 64x64 tiles.
//!
//! Tiles live in a fixed GPU pool. `sparse_life.wgsl` steps every active tile and
//! reports which tiles are alive and which neighbours they need; every
//! `tiles::MARGIN` generations the CPU reads those flags back and allocates or
//! frees tiles (`tiles.rs`). `sparse_render.wgsl` draws the world one cell per
//! screen pixel, centred on the origin.

use std::collections::HashMap;

use crate::engine::{BYTE_UNITS, Engine, GpuContext, bind, fullscreen_pipeline, human, layout_entry};
use crate::patterns;
use crate::tiles::{self, MARGIN, TILE, TILE_CELLS, TILE_WORDS, TileWorld};

/// Bytes for one tile in one cell buffer (bit-packed: 128 words = 512 bytes).
const TILE_BYTES: u64 = (TILE_WORDS * 4) as u64;
/// Upper bound on each of the two cell buffers. The real pool may be smaller if the
/// GPU's buffer limits are lower. 512 MiB = 1,048,576 tiles = ~4.3 billion cells.
const POOL_BUDGET_BYTES: u64 = 512 << 20;
/// Largest tile map the renderer supports (covers 16384 x 16384 pixels).
const MAX_VIEW_TILES: u32 = 256;
/// Side length of the random soup square.
const RANDOM_SIZE: i32 = 1000;

/// Compute shader parameters.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    count: u32,
}

/// Render shader parameters; matches `View` in sparse_render.wgsl.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct View {
    screen_origin: [i32; 2],
    map_origin: [i32; 2],
    map_size: [u32; 2],
    show_tiles: u32,
    _pad: u32,
}

pub struct SparseEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    step_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    params_buffer: wgpu::Buffer,
    active_buffer: wgpu::Buffer,
    neighbor_buffer: wgpu::Buffer,
    cell_buffers: [wgpu::Buffer; 2],
    flags_buffer: wgpu::Buffer,
    flags_staging: wgpu::Buffer,
    view_buffer: wgpu::Buffer,
    tile_map_buffer: wgpu::Buffer,
    /// `compute_bind_groups[i]` reads cell buffer `i` and writes buffer `1 - i`.
    compute_bind_groups: [wgpu::BindGroup; 2],
    /// `render_bind_groups[i]` draws cell buffer `i`.
    render_bind_groups: [wgpu::BindGroup; 2],
    /// Index of the cell buffer that holds the current generation.
    current: usize,

    world: TileWorld,
    /// Generations run since the last bookkeeping pass (always < MARGIN).
    since_bookkeeping: u32,
    zero_tile: Vec<u8>,
    last_view: Option<(View, Vec<u32>)>,
    show_tiles: bool,
    generation: u64,
    /// Live cells as of the last bookkeeping pass, and that pass's generation.
    population: (u64, u64),
}

impl SparseEngine {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;

        // Size the tile pool from the device's buffer limit, capped by our budget.
        let capacity = (POOL_BUDGET_BYTES.min(ctx.max_buffer) / TILE_BYTES) as u32;
        let pool_bytes = u64::from(capacity) * TILE_BYTES;
        log::info!(
            "Sparse tile pool: {capacity} tiles of {TILE}x{TILE} ({} MiB per buffer, 2 buffers)",
            pool_bytes >> 20
        );

        let storage = |label: &str, size: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        let uniform = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cap = u64::from(capacity);
        let dst = wgpu::BufferUsages::COPY_DST;
        let params_buffer = uniform("sparse params", size_of::<Params>() as u64);
        let active_buffer = storage("active tiles", cap * 4, dst);
        let neighbor_buffer = storage("tile neighbours", cap * 32, dst);
        let cell_buffers = [0, 1].map(|i| storage(&format!("tile cells {i}"), pool_bytes, dst));
        let flags_buffer = storage("tile flags", cap * 4, wgpu::BufferUsages::COPY_SRC);
        let flags_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile flags readback"),
            size: cap * 4,
            usage: wgpu::BufferUsages::MAP_READ | dst,
            mapped_at_creation: false,
        });
        let view_buffer = uniform("sparse view", size_of::<View>() as u64);
        let tile_map_buffer =
            storage("view tile map", u64::from(MAX_VIEW_TILES * MAX_VIEW_TILES) * 4, dst);

        // --- Compute pipeline ----------------------------------------------

        use wgpu::BufferBindingType::{Storage, Uniform};
        let cs = wgpu::ShaderStages::COMPUTE;
        let life_shader = device.create_shader_module(wgpu::include_wgsl!("sparse_life.wgsl"));
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sparse compute layout"),
            entries: &[
                layout_entry(0, cs, Uniform),
                layout_entry(1, cs, Storage { read_only: true }),
                layout_entry(2, cs, Storage { read_only: true }),
                layout_entry(3, cs, Storage { read_only: true }),
                layout_entry(4, cs, Storage { read_only: false }),
                layout_entry(5, cs, Storage { read_only: false }),
            ],
        });
        let step_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sparse step"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sparse compute pipeline layout"),
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
                label: Some(&format!("sparse compute {i} -> {}", 1 - i)),
                layout: &compute_layout,
                entries: &[
                    bind(0, &params_buffer),
                    bind(1, &active_buffer),
                    bind(2, &neighbor_buffer),
                    bind(3, &cell_buffers[i]),
                    bind(4, &cell_buffers[1 - i]),
                    bind(5, &flags_buffer),
                ],
            })
        });

        // --- Render pipeline -----------------------------------------------

        let fs = wgpu::ShaderStages::FRAGMENT;
        let render_shader = device.create_shader_module(wgpu::include_wgsl!("sparse_render.wgsl"));
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sparse render layout"),
            entries: &[
                layout_entry(0, fs, Uniform),
                layout_entry(1, fs, Storage { read_only: true }),
                layout_entry(2, fs, Storage { read_only: true }),
            ],
        });
        let render_pipeline = fullscreen_pipeline(ctx, "sparse draw", &render_shader, &render_layout);
        let render_bind_groups = [0, 1].map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("sparse render {i}")),
                layout: &render_layout,
                entries: &[
                    bind(0, &view_buffer),
                    bind(1, &tile_map_buffer),
                    bind(2, &cell_buffers[i]),
                ],
            })
        });

        Self {
            device: ctx.device.clone(),
            queue: ctx.queue.clone(),
            step_pipeline,
            render_pipeline,
            params_buffer,
            active_buffer,
            neighbor_buffer,
            cell_buffers,
            flags_buffer,
            flags_staging,
            view_buffer,
            tile_map_buffer,
            compute_bind_groups,
            render_bind_groups,
            current: 0,
            world: TileWorld::new(capacity),
            since_bookkeeping: 0,
            zero_tile: vec![0; TILE_BYTES as usize],
            last_view: None,
            show_tiles: false,
            generation: 0,
            population: (0, 0),
        }
    }

    /// Number of allocated tiles.
    pub fn tile_count(&self) -> usize {
        self.world.len()
    }

    /// Upload the tile world's pending changes to the GPU.
    fn apply_world_changes(&mut self) {
        let changes = self.world.take_changes();
        for (slot, row) in &changes.neighbors {
            self.queue.write_buffer(
                &self.neighbor_buffer,
                u64::from(*slot) * 32,
                bytemuck::cast_slice(row),
            );
        }
        for slot in &changes.new_slots {
            self.queue.write_buffer(
                &self.cell_buffers[self.current],
                u64::from(*slot) * TILE_BYTES,
                &self.zero_tile,
            );
        }
        if let Some(active) = &changes.active {
            if !active.is_empty() {
                self.queue.write_buffer(&self.active_buffer, 0, bytemuck::cast_slice(active));
            }
            let params = Params { count: active.len() as u32 };
            self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        }
    }

    /// Run `n` generations (n <= MARGIN - since_bookkeeping) in one submission.
    fn run_generations(&mut self, n: u32) {
        let count = self.world.active().len() as u32;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if count > 0 {
            // Spread tiles over a 2D dispatch to stay under the 65535 per-axis limit.
            let x = count.min(65535);
            let y = count.div_ceil(x);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.step_pipeline);
            for i in 0..n as usize {
                let src = (self.current + i) % 2;
                pass.set_bind_group(0, &self.compute_bind_groups[src], &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
        self.queue.submit([encoder.finish()]);
        if n % 2 == 1 {
            self.current = 1 - self.current;
        }
        self.since_bookkeeping += n;
        self.generation += u64::from(n);
    }

    /// Read back tile flags and grow / shrink the world to match.
    fn bookkeeping(&mut self) {
        let count = self.world.active().len();
        let flags: Vec<u32> = if count == 0 {
            Vec::new()
        } else {
            let bytes = (count * 4) as u64;
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.flags_buffer, 0, &self.flags_staging, 0, bytes);
            self.queue.submit([encoder.finish()]);
            self.flags_staging.map_async(wgpu::MapMode::Read, 0..bytes, |r| {
                r.expect("map tile flags");
            });
            self.device.poll(wgpu::PollType::wait_indefinitely()).expect("wait for GPU");
            let flags = {
                let view = self.flags_staging.get_mapped_range(0..bytes).expect("mapped flags");
                bytemuck::cast_slice::<u8, u32>(&view).to_vec()
            };
            self.flags_staging.unmap();
            flags
        };
        let population = flags.iter().map(|f| u64::from(f >> tiles::POP_SHIFT)).sum();
        self.population = (population, self.generation);
        self.world.update(&flags);
        self.apply_world_changes();
        self.since_bookkeeping = 0;
    }
}

impl Engine for SparseEngine {
    fn title(&self) -> String {
        "ENGINE SPARSE: BIT-PACKED 64X64 TILES, UNBOUNDED".to_owned()
    }

    fn info(&self) -> String {
        let used = self.world.len() as u64;
        let capacity = u64::from(self.world.capacity());
        let pct = 100.0 * used as f64 / capacity.max(1) as f64;
        // Both cell buffers count: each tile costs 2 x TILE_BYTES.
        let bytes = |tiles: u64| human((tiles * 2 * TILE_BYTES) as f64, BYTE_UNITS);
        let full = if self.world.overflowed { "  POOL FULL" } else { "" };
        format!(
            "TILES {used} OF {capacity}  {pct:.1}% USED{full}\n\
             TILE MEMORY {} OF {}",
            bytes(used),
            bytes(capacity)
        )
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn population(&self) -> Option<(u64, u64)> {
        Some(self.population)
    }

    fn cells_per_generation(&self) -> u64 {
        self.world.len() as u64 * TILE_CELLS as u64
    }

    fn load(&mut self, cells: &[(i32, i32)]) {
        self.world.clear();
        let mut tile_data: HashMap<tiles::TileCoord, Vec<u32>> = HashMap::new();
        for &(x, y) in cells {
            let (tile, i) = tiles::locate(x, y);
            tile_data.entry(tile).or_insert_with(|| vec![0; TILE_WORDS])[i / 32] |= 1 << (i % 32);
        }
        let mut placed = Vec::new();
        for (tile, data) in tile_data {
            if let Some(slot) = self.world.allocate(tile) {
                placed.push((slot, data));
            }
        }

        // Run a bookkeeping pass with CPU-computed flags so the halo tiles exist.
        self.world.sync_active();
        let by_slot: HashMap<u32, &Vec<u32>> = placed.iter().map(|(s, d)| (*s, d)).collect();
        let flags: Vec<u32> = self
            .world
            .active()
            .iter()
            .map(|s| by_slot.get(s).map_or(0, |d| tiles::flags_for(d)))
            .collect();
        self.world.update(&flags);

        // Zero new tiles first, then write the pattern (queue writes run in order).
        self.apply_world_changes();
        for (slot, data) in &placed {
            self.queue.write_buffer(
                &self.cell_buffers[self.current],
                u64::from(*slot) * TILE_BYTES,
                bytemuck::cast_slice(data),
            );
        }
        self.since_bookkeeping = 0;
        self.generation = 0;
        let population = flags.iter().map(|f| u64::from(f >> tiles::POP_SHIFT)).sum();
        self.population = (population, 0);
    }

    fn random(&mut self) {
        self.load(&patterns::random_square(RANDOM_SIZE, crate::engine::time_seed()));
    }

    fn step(&mut self, n: u32) {
        // Run in batches that never cross a bookkeeping boundary.
        let mut remaining = n;
        while remaining > 0 {
            let batch = remaining.min(MARGIN - self.since_bookkeeping);
            self.run_generations(batch);
            remaining -= batch;
            if self.since_bookkeeping == MARGIN {
                self.bookkeeping();
            }
        }
    }

    fn update(&mut self, screen_width: u32, screen_height: u32) {
        let (w, h) = (screen_width as i32, screen_height as i32);
        // Centre the world origin in the window.
        let screen_origin = [-(w / 2), -(h / 2)];
        let t = TILE as i32;
        let first = [screen_origin[0].div_euclid(t), screen_origin[1].div_euclid(t)];
        let last = [
            (screen_origin[0] + w - 1).div_euclid(t),
            (screen_origin[1] + h - 1).div_euclid(t),
        ];
        let cols = ((last[0] - first[0] + 1) as u32).min(MAX_VIEW_TILES);
        let rows = ((last[1] - first[1] + 1) as u32).min(MAX_VIEW_TILES);

        let mut map = Vec::with_capacity((cols * rows) as usize);
        for ty in 0..rows as i32 {
            for tx in 0..cols as i32 {
                let slot = self.world.slot((first[0] + tx, first[1] + ty));
                map.push(slot.unwrap_or(tiles::EMPTY));
            }
        }
        let view = View {
            screen_origin,
            map_origin: first,
            map_size: [cols, rows],
            show_tiles: u32::from(self.show_tiles),
            _pad: 0,
        };
        if self.last_view.as_ref() != Some(&(view, map.clone())) {
            self.queue.write_buffer(&self.view_buffer, 0, bytemuck::bytes_of(&view));
            self.queue.write_buffer(&self.tile_map_buffer, 0, bytemuck::cast_slice(&map));
            self.last_view = Some((view, map));
        }
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_bind_groups[self.current], &[]);
        pass.draw(0..3, 0..1);
    }

    fn toggle_tiles(&mut self) {
        self.show_tiles = !self.show_tiles;
    }
}
