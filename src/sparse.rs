//! Sparse engine: an effectively unbounded world made of TILE x TILE tiles.
//!
//! Tiles live in a fixed GPU pool. `sparse_life.wgsl` steps every active tile and
//! reports which tiles are alive and which neighbours they need; every
//! `tiles::BOOKKEEPING_INTERVAL` generations the CPU reads those flags back and
//! allocates or frees tiles (`tiles.rs`). `sparse_render.wgsl` draws the world one
//! cell per screen pixel, centred on the origin.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::engine::{
    BYTE_UNITS, Engine, GpuContext, LIFE_RULE_WGSL, bind, fullscreen_pipeline, human, layout_entry,
    readback_buffer, shader, storage_buffer, uniform_buffer,
};
use crate::patterns;
use crate::tiles::{self, BOOKKEEPING_INTERVAL, TILE, TILE_CELLS, TILE_WORDS, TileCoord, TileWorld};

/// Bytes for one tile in one cell buffer (bit-packed, one bit per cell).
const TILE_BYTES: u64 = (TILE_WORDS * 4) as u64;
/// Upper bound on each of the two cell buffers. The real pool may be smaller if the
/// GPU's buffer limits are lower. 512 MiB of bits = ~4.3 billion cells.
const POOL_BUDGET_BYTES: u64 = 512 << 20;
/// Largest tile map the renderer supports, in tiles per side.
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

/// GPU buffers, all sized for `capacity` tiles.
struct Buffers {
    params: wgpu::Buffer,
    /// Slot of each active tile, in the order the GPU processes them.
    active: wgpu::Buffer,
    /// 8 neighbour slots per tile slot.
    neighbors: wgpu::Buffer,
    /// The tile pool, ping-ponged each generation.
    cells: [wgpu::Buffer; 2],
    /// Per-active-tile flags written by the compute shader.
    flags: wgpu::Buffer,
    view: wgpu::Buffer,
    /// Slot of each tile on screen, for the renderer.
    tile_map: wgpu::Buffer,
}

impl Buffers {
    fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let cap = u64::from(capacity);
        let dst = wgpu::BufferUsages::COPY_DST;
        let pool_bytes = cap * TILE_BYTES;
        Self {
            params: uniform_buffer(device, "sparse params", size_of::<Params>() as u64),
            active: storage_buffer(device, "active tiles", cap * 4, dst),
            neighbors: storage_buffer(device, "tile neighbours", cap * 32, dst),
            cells: [0, 1].map(|i| storage_buffer(device, &format!("tile cells {i}"), pool_bytes, dst)),
            flags: storage_buffer(device, "tile flags", cap * 4, wgpu::BufferUsages::COPY_SRC),
            view: uniform_buffer(device, "sparse view", size_of::<View>() as u64),
            tile_map: storage_buffer(
                device,
                "view tile map",
                u64::from(MAX_VIEW_TILES * MAX_VIEW_TILES) * 4,
                dst,
            ),
        }
    }
}

/// Pipelines and bind groups. `bind_groups[i]` reads (compute) or draws (render)
/// cell buffer `i`; compute writes buffer `1 - i`.
struct Pipeline<P> {
    pipeline: P,
    bind_groups: [wgpu::BindGroup; 2],
}

/// Tile flags read back from the GPU, and which tiles they describe.
struct Flags {
    values: Vec<u32>,
    /// Tile of each flag, in order (slots may have been reused since).
    coords: Vec<TileCoord>,
    generation: u64,
}

/// Pipelined flags readback: each bookkeeping point starts copying the current
/// flags into one staging buffer and collects the previous point's copy from the
/// other, so the GPU never waits on the CPU. See tiles.rs for why applying flags
/// one pass late is safe.
struct FlagReadback {
    staging: [wgpu::Buffer; 2],
    in_flight: Option<InFlight>,
}

struct InFlight {
    staging: usize,
    submission: wgpu::SubmissionIndex,
    coords: Vec<TileCoord>,
    generation: u64,
}

impl FlagReadback {
    fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let bytes = u64::from(capacity) * 4;
        Self {
            staging: [0, 1].map(|i| readback_buffer(device, &format!("tile flags readback {i}"), bytes)),
            in_flight: None,
        }
    }

    /// Start reading back `flags` (one per tile in `coords`, as of `generation`),
    /// and return the previous readback's result, if there was one.
    fn swap(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        flags: &wgpu::Buffer,
        coords: Vec<TileCoord>,
        generation: u64,
    ) -> Option<Flags> {
        let previous = self.in_flight.take();
        if !coords.is_empty() {
            let staging = previous.as_ref().map_or(0, |p| 1 - p.staging);
            let bytes = (coords.len() * 4) as u64;
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(flags, 0, &self.staging[staging], 0, bytes);
            let submission = queue.submit([encoder.finish()]);
            self.staging[staging].map_async(wgpu::MapMode::Read, 0..bytes, |r| {
                r.expect("map tile flags");
            });
            self.in_flight = Some(InFlight { staging, submission, coords, generation });
        }
        previous.map(|p| self.collect(device, p))
    }

    /// Drop any in-flight readback (e.g. when the world is replaced).
    fn discard(&mut self, device: &wgpu::Device) {
        if let Some(in_flight) = self.in_flight.take() {
            self.collect(device, in_flight);
        }
    }

    /// Wait for a readback (normally long finished) and take its flags.
    fn collect(&self, device: &wgpu::Device, in_flight: InFlight) -> Flags {
        // Wait for that submission only, not for everything queued since.
        device
            .poll(wgpu::PollType::Wait { submission_index: Some(in_flight.submission), timeout: None })
            .expect("wait for tile flags");
        let buffer = &self.staging[in_flight.staging];
        let bytes = (in_flight.coords.len() * 4) as u64;
        let values = {
            let view = buffer.get_mapped_range(0..bytes).expect("mapped flags");
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        buffer.unmap();
        Flags { values, coords: in_flight.coords, generation: in_flight.generation }
    }
}

pub struct SparseEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: Buffers,
    compute: Pipeline<wgpu::ComputePipeline>,
    render: Pipeline<wgpu::RenderPipeline>,
    readback: FlagReadback,
    /// Index of the cell buffer that holds the current generation.
    current_buffer: usize,

    world: TileWorld,
    /// Generations run since the last bookkeeping pass (always < BOOKKEEPING_INTERVAL).
    since_bookkeeping: u32,
    zero_tile: Vec<u8>,
    last_view: Option<(View, Vec<u32>)>,
    show_tiles: bool,
    generation: u64,
    /// Live cells as of the last applied readback, and that readback's generation.
    population: (u64, u64),
    /// Total CPU time spent applying readbacks (for the benchmark).
    bookkeeping_time: Duration,
}

impl SparseEngine {
    pub fn new(ctx: &GpuContext) -> Self {
        // Size the tile pool from the device's buffer limit, capped by our budget.
        let capacity = (POOL_BUDGET_BYTES.min(ctx.max_buffer) / TILE_BYTES) as u32;
        log::info!(
            "Sparse tile pool: {capacity} tiles of {TILE}x{TILE} ({} MiB per buffer, 2 buffers)",
            (u64::from(capacity) * TILE_BYTES) >> 20
        );
        let buffers = Buffers::new(&ctx.device, capacity);
        Self {
            device: ctx.device.clone(),
            queue: ctx.queue.clone(),
            compute: create_compute(ctx, &buffers),
            render: create_render(ctx, &buffers),
            readback: FlagReadback::new(&ctx.device, capacity),
            buffers,
            current_buffer: 0,
            world: TileWorld::new(capacity),
            since_bookkeeping: 0,
            zero_tile: vec![0; TILE_BYTES as usize],
            last_view: None,
            show_tiles: false,
            generation: 0,
            population: (0, 0),
            bookkeeping_time: Duration::ZERO,
        }
    }

    /// CPU time spent applying readbacks so far, and reset the counter.
    pub fn take_bookkeeping_time(&mut self) -> Duration {
        std::mem::take(&mut self.bookkeeping_time)
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
                &self.buffers.neighbors,
                u64::from(*slot) * 32,
                bytemuck::cast_slice(row),
            );
        }
        for slot in &changes.new_slots {
            self.queue.write_buffer(
                &self.buffers.cells[self.current_buffer],
                u64::from(*slot) * TILE_BYTES,
                &self.zero_tile,
            );
        }
        if let Some(active) = &changes.active {
            if !active.is_empty() {
                self.queue.write_buffer(&self.buffers.active, 0, bytemuck::cast_slice(active));
            }
            let params = Params { count: active.len() as u32 };
            self.queue.write_buffer(&self.buffers.params, 0, bytemuck::bytes_of(&params));
        }
    }

    /// Run `n` generations (n <= BOOKKEEPING_INTERVAL - since_bookkeeping) in one
    /// submission.
    fn run_generations(&mut self, n: u32) {
        let count = self.world.active().len() as u32;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if count > 0 {
            // Spread tiles over a 2D dispatch to stay under the 65535 per-axis limit.
            let x = count.min(65535);
            let y = count.div_ceil(x);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.compute.pipeline);
            for i in 0..n as usize {
                let src = (self.current_buffer + i) % 2;
                pass.set_bind_group(0, &self.compute.bind_groups[src], &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
        self.queue.submit([encoder.finish()]);
        if n % 2 == 1 {
            self.current_buffer = 1 - self.current_buffer;
        }
        self.since_bookkeeping += n;
        self.generation += u64::from(n);
    }

    /// Bookkeeping point: start reading back the current flags, and grow / shrink
    /// the world from the previous point's readback.
    fn bookkeeping(&mut self) {
        let coords = self.world.active_coords();
        let previous =
            self.readback.swap(&self.device, &self.queue, &self.buffers.flags, coords, self.generation);
        if let Some(flags) = previous {
            let population = flags.values.iter().map(|f| u64::from(f >> tiles::POP_SHIFT)).sum();
            self.population = (population, flags.generation);
            let start = Instant::now();
            self.world.update_from(&flags.coords, &flags.values);
            self.apply_world_changes();
            self.bookkeeping_time += start.elapsed();
        }
        self.since_bookkeeping = 0;
    }
}

fn create_compute(ctx: &GpuContext, buffers: &Buffers) -> Pipeline<wgpu::ComputePipeline> {
    use wgpu::BufferBindingType::{Storage, Uniform};
    let device = &ctx.device;
    let cs = wgpu::ShaderStages::COMPUTE;
    let module = shader(
        device,
        "sparse_life.wgsl",
        &[&tiles::shader_constants(), LIFE_RULE_WGSL, include_str!("sparse_life.wgsl")],
    );
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sparse step"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sparse compute pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        })),
        module: &module,
        entry_point: Some("step"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_groups = [0, 1].map(|i| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("sparse compute {i} -> {}", 1 - i)),
            layout: &layout,
            entries: &[
                bind(0, &buffers.params),
                bind(1, &buffers.active),
                bind(2, &buffers.neighbors),
                bind(3, &buffers.cells[i]),
                bind(4, &buffers.cells[1 - i]),
                bind(5, &buffers.flags),
            ],
        })
    });
    Pipeline { pipeline, bind_groups }
}

fn create_render(ctx: &GpuContext, buffers: &Buffers) -> Pipeline<wgpu::RenderPipeline> {
    use wgpu::BufferBindingType::{Storage, Uniform};
    let device = &ctx.device;
    let fs = wgpu::ShaderStages::FRAGMENT;
    let module = shader(
        device,
        "sparse_render.wgsl",
        &[&tiles::shader_constants(), include_str!("sparse_render.wgsl")],
    );
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sparse render layout"),
        entries: &[
            layout_entry(0, fs, Uniform),
            layout_entry(1, fs, Storage { read_only: true }),
            layout_entry(2, fs, Storage { read_only: true }),
        ],
    });
    let bind_groups = [0, 1].map(|i| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("sparse render {i}")),
            layout: &layout,
            entries: &[
                bind(0, &buffers.view),
                bind(1, &buffers.tile_map),
                bind(2, &buffers.cells[i]),
            ],
        })
    });
    Pipeline { pipeline: fullscreen_pipeline(ctx, "sparse draw", &module, &layout), bind_groups }
}

impl Engine for SparseEngine {
    fn title(&self) -> String {
        format!("ENGINE SPARSE: BIT-PACKED {TILE}X{TILE} TILES, UNBOUNDED")
    }

    fn info(&self) -> Vec<String> {
        let used = self.world.len() as u64;
        let capacity = u64::from(self.world.capacity());
        let pct = 100.0 * used as f64 / capacity.max(1) as f64;
        // Both cell buffers count: each tile costs 2 x TILE_BYTES.
        let bytes = |tiles: u64| human((tiles * 2 * TILE_BYTES) as f64, BYTE_UNITS);
        let full = if self.world.overflowed { "  POOL FULL" } else { "" };
        vec![
            format!("TILES {used} OF {capacity}  {pct:.1}% USED{full}"),
            format!("TILE MEMORY {} OF {}", bytes(used), bytes(capacity)),
        ]
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
        self.readback.discard(&self.device);
        self.world.clear();
        let mut tile_data: HashMap<TileCoord, Vec<u32>> = HashMap::new();
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
                &self.buffers.cells[self.current_buffer],
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
            let batch = remaining.min(BOOKKEEPING_INTERVAL - self.since_bookkeeping);
            self.run_generations(batch);
            remaining -= batch;
            if self.since_bookkeeping == BOOKKEEPING_INTERVAL {
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
            self.queue.write_buffer(&self.buffers.view, 0, bytemuck::bytes_of(&view));
            self.queue.write_buffer(&self.buffers.tile_map, 0, bytemuck::cast_slice(&map));
            self.last_view = Some((view, map));
        }
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.render.pipeline);
        pass.set_bind_group(0, &self.render.bind_groups[self.current_buffer], &[]);
        pass.draw(0..3, 0..1);
    }

    fn toggle_tiles(&mut self) {
        self.show_tiles = !self.show_tiles;
    }
}
