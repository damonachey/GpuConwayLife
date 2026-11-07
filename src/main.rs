//! GPU Conway's Game of Life with two swappable engines.
//!
//! - **Dense** (`dense.rs`): one huge bit-packed grid with dead edges.
//! - **Sparse** (`sparse.rs`): an unbounded world of bit-packed tiles (size set by
//!   `tiles::TILE`) that grows and shrinks with the life in it.
//!
//! This file holds what they share: the window, keyboard, speed control, stats and
//! the text overlay. Pressing E drops the current engine (freeing its GPU memory),
//! builds the other one and restarts whatever was last loaded.
//!
//! Controls: Space pause/resume, N step (while paused), R random soup, 1 glider gun,
//! 2 acorn, 3 bunnies, 4 Ark1, E switch engine, T show tiles (sparse), +/- speed.

mod bench;
mod dense;
mod engine;
mod overlay;
mod patterns;
mod sparse;
mod tiles;

use std::sync::Arc;
use std::time::{Duration, Instant};

use engine::{Engine, GpuContext, human};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Upper bound on any single GPU buffer (most GPUs allow ~2 GiB per binding).
const MAX_BUFFER_BYTES: u64 = 2 << 30;
/// Speed is 2^level generations per frame, so -4 is one generation every 16 frames.
const MIN_SPEED_LEVEL: i32 = -4;
const MAX_SPEED_LEVEL: i32 = 10;
/// Cap on simulation work per frame, so big worlds at high speed stay responsive.
/// Generations per frame are limited to what the GPU did in this much time recently.
const FRAME_BUDGET: Duration = Duration::from_millis(100);
/// How often FPS and generations/second are recalculated.
const STATS_PERIOD: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, PartialEq, Debug)]
enum EngineKind {
    Dense,
    Sparse,
}

/// What the world was last started from, so an engine switch can restart it.
#[derive(Clone, Copy)]
enum Start {
    Random,
    Pattern(&'static [&'static str]),
}

/// Speed, pausing and single-stepping, controlled from the keyboard.
struct Pacing {
    paused: bool,
    speed_level: i32,
    /// Fractional generations owed, for speeds below one per frame.
    pending: f64,
    /// Generations requested with N while paused.
    single_steps: u32,
}

impl Pacing {
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
    fn new() -> Self {
        Self { sample_start: Instant::now(), frames: 0, generations: 0, fps: 0.0, gens_per_sec: 0.0 }
    }

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

struct App {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    ctx: GpuContext,
    /// Always present, except for the moment during an engine switch when the old
    /// one has been dropped (to free its GPU memory) and the new one isn't built yet.
    engine: Option<Box<dyn Engine>>,
    kind: EngineKind,
    start: Start,
    overlay: overlay::Overlay,
    pacing: Pacing,
    stats: Stats,
}

impl App {
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
        let (device, queue, max_buffer) = create_device(&adapter).await;

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by adapter");
        surface.configure(&device, &config);

        let ctx = GpuContext { device, queue, format: config.format, max_buffer };
        let mut overlay = overlay::Overlay::new(&ctx.device, config.format);
        overlay.resize(config.width, config.height, window.scale_factor());

        let kind = EngineKind::Dense;
        let mut engine = build_engine(&ctx, kind).await;
        engine.random();

        Self {
            window,
            surface,
            config,
            ctx,
            engine: Some(engine),
            kind,
            start: Start::Random,
            overlay,
            pacing: Pacing { paused: false, speed_level: 0, pending: 0.0, single_steps: 0 },
            stats: Stats::new(),
        }
    }

    fn engine(&self) -> &dyn Engine {
        self.engine.as_deref().expect("engine present outside a switch")
    }

    fn engine_mut(&mut self) -> &mut dyn Engine {
        self.engine.as_deref_mut().expect("engine present outside a switch")
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.ctx.device, &self.config);
            self.overlay.resize(width, height, self.window.scale_factor());
        }
    }

    fn key(&mut self, event: &KeyEvent) {
        log::debug!("key {:?} {:?} repeat={}", event.logical_key, event.state, event.repeat);
        if event.state != ElementState::Pressed {
            return;
        }
        let key = match &event.logical_key {
            Key::Named(NamedKey::Space) => ' ',
            Key::Character(text) => match text.chars().next() {
                Some(c) => c.to_ascii_lowercase(),
                None => return,
            },
            _ => return,
        };
        // Most keys act once per press; N and +/- also repeat while held.
        let fresh = !event.repeat;
        let pacing = &mut self.pacing;
        match key {
            ' ' if fresh => {
                pacing.paused = !pacing.paused;
                pacing.pending = 0.0;
            }
            'n' if pacing.paused => pacing.single_steps += 1,
            '+' | '=' => pacing.speed_level = (pacing.speed_level + 1).min(MAX_SPEED_LEVEL),
            '-' | '_' => pacing.speed_level = (pacing.speed_level - 1).max(MIN_SPEED_LEVEL),
            'e' if fresh => self.switch_engine(),
            't' if fresh => self.engine_mut().toggle_tiles(),
            'r' if fresh => self.restart(Start::Random),
            c if fresh => {
                if let Some(&(_, _, pattern)) = patterns::KEYED.iter().find(|(k, ..)| *k == c) {
                    self.restart(Start::Pattern(pattern));
                }
            }
            _ => {}
        }
    }

    /// Load `start` into the current engine and remember it for engine switches.
    fn restart(&mut self, start: Start) {
        self.start = start;
        match start {
            Start::Random => self.engine_mut().random(),
            Start::Pattern(pattern) => self.engine_mut().load(&patterns::centered(pattern)),
        }
        self.pacing.pending = 0.0;
    }

    /// Swap to the other engine and restart the current pattern in it.
    fn switch_engine(&mut self) {
        let next = match self.kind {
            EngineKind::Dense => EngineKind::Sparse,
            EngineKind::Sparse => EngineKind::Dense,
        };
        log::info!("Switching engine: {:?} -> {next:?}", self.kind);
        // Free the old engine's GPU memory before allocating the new one's.
        self.engine = None;
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
        self.engine = Some(pollster::block_on(build_engine(&self.ctx, next)));
        self.kind = next;
        // Speeds don't carry over meaningfully between engines.
        self.stats = Stats::new();
        self.restart(self.start);
    }

    fn overlay_text(&self) -> String {
        let (pacing, stats, engine) = (&self.pacing, &self.stats, self.engine());
        let ms = if stats.fps > 0.0 { 1000.0 / stats.fps } else { 0.0 };
        let paused = if pacing.paused { "  PAUSED" } else { "" };
        let cell_rate = human(stats.gens_per_sec * engine.cells_per_generation() as f64, &[
            (1e12, "TRILLION"),
            (1e9, "BILLION"),
            (1e6, "MILLION"),
        ]);
        let population = match engine.population() {
            Some((cells, generation)) => format!("POP {cells} AT GEN {generation}"),
            None => "POP COUNTING".to_owned(),
        };
        let pattern_keys: Vec<String> =
            patterns::KEYED.iter().map(|(key, label, _)| format!("{key} {label}")).collect();

        let mut lines = vec![
            engine.title(),
            format!("{:.0} FPS  {ms:.2} MS", stats.fps),
            format!(
                "GEN {}  {:.0} GEN/S  {cell_rate} CELLS/S",
                engine.generation(),
                stats.gens_per_sec
            ),
            population,
            format!("SPEED {} GEN/FRAME{paused}", pacing.speed_label()),
        ];
        lines.extend(engine.info());
        lines.push("SPACE PAUSE  N STEP  R RANDOM  +/- SPEED  E SWITCH ENGINE".to_owned());
        lines.push(format!("{}  T SHOW TILES", pattern_keys.join("  ")));
        lines.join("\n")
    }

    /// Advance the simulation as needed and draw a frame.
    fn frame(&mut self) {
        use wgpu::CurrentSurfaceTexture as Cst;
        let device = &self.ctx.device;
        let frame = match self.surface.get_current_texture() {
            Cst::Success(frame) => frame,
            Cst::Suboptimal(frame) => {
                // Draw this one, but reconfigure so the next frame is a perfect fit.
                self.surface.configure(device, &self.config);
                frame
            }
            Cst::Outdated | Cst::Lost => {
                self.surface.configure(device, &self.config);
                return;
            }
            Cst::Timeout | Cst::Occluded => return,
            Cst::Validation => {
                log::warn!("surface validation error; skipping frame");
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());

        let mut generations = self.pacing.take_generations();
        if !self.pacing.paused {
            // Don't queue more than ~FRAME_BUDGET of GPU work, judged by recent
            // throughput; if the GPU can't keep up with the requested speed, drop
            // the excess rather than falling further behind.
            let budget = (self.stats.gens_per_sec * FRAME_BUDGET.as_secs_f64()).max(1.0) as u32;
            generations = generations.min(budget);
        }
        self.engine_mut().step(generations);
        self.stats.record(generations);
        let (width, height) = (self.config.width, self.config.height);
        self.engine_mut().update(width, height);
        let text = self.overlay_text();
        self.overlay.set_text(&text);

        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
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
            self.engine.as_deref().expect("engine present").draw(&mut pass);
            self.overlay.draw(&self.ctx.queue, &mut pass);
        }

        self.ctx.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.ctx.queue.present(frame);
    }
}

/// Create the device, asking for the biggest buffers the GPU supports (wgpu
/// defaults to 128 MiB). Returns the device, queue and usable max buffer size.
async fn create_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue, u64) {
    let adapter_limits = adapter.limits();
    let max_buffer = MAX_BUFFER_BYTES
        .min(adapter_limits.max_storage_buffer_binding_size)
        .min(adapter_limits.max_buffer_size);
    let defaults = wgpu::Limits::default();
    let required_limits = wgpu::Limits {
        max_storage_buffer_binding_size: max_buffer.max(defaults.max_storage_buffer_binding_size),
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
    (device, queue, max_buffer)
}

/// `--bench`: run the headless benchmark instead of opening a window.
fn run_bench() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no suitable GPU adapter");
    let info = adapter.get_info();
    println!("Adapter: {} ({:?})", info.name, info.backend);
    let (device, queue, max_buffer) = pollster::block_on(create_device(&adapter));
    // Any render format works; nothing is presented.
    let ctx =
        GpuContext { device, queue, format: wgpu::TextureFormat::Bgra8UnormSrgb, max_buffer };
    bench::run(&ctx);
}

async fn build_engine(ctx: &GpuContext, kind: EngineKind) -> Box<dyn Engine> {
    match kind {
        EngineKind::Dense => Box::new(dense::DenseEngine::new(ctx).await),
        EngineKind::Sparse => Box::new(sparse::SparseEngine::new(ctx)),
    }
}

/// Receives winit's events and creates the `App` once the window can exist.
#[derive(Default)]
struct WinitHandler {
    app: Option<App>,
}

impl ApplicationHandler for WinitHandler {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_some() {
            return;
        }
        // Physical size so one cell maps to exactly one screen pixel.
        let attrs = Window::default_attributes()
            .with_title("GPU Conway's Game of Life")
            .with_inner_size(winit::dpi::PhysicalSize::new(1000, 1000));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.app = Some(pollster::block_on(App::new(window)));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(app) = self.app.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => app.resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => app.key(&event),
            WindowEvent::RedrawRequested => app.frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(app) = &self.app {
            app.window.request_redraw();
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
    if std::env::args().any(|a| a == "--bench") {
        run_bench();
        return;
    }
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut WinitHandler::default()).expect("run event loop");
}
