//! GPU Conway's Game of Life.
//!
//! The simulation runs in a wgpu compute shader (`life.wgsl`) and is drawn by a
//! fragment shader (`render.wgsl`) that reads the same GPU buffer. The CPU only
//! seeds the first generation; after that it just submits commands.

mod overlay;

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const GRID_WIDTH: u32 = 256;
const GRID_HEIGHT: u32 = 256;
const WORKGROUP_SIZE: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
}

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    /// `compute_bind_groups[i]` reads buffer `i` and writes buffer `1 - i`.
    compute_bind_groups: [wgpu::BindGroup; 2],
    /// `render_bind_groups[i]` draws buffer `i`.
    render_bind_groups: [wgpu::BindGroup; 2],
    /// Index of the buffer that holds the current generation.
    current: usize,
    overlay: overlay::Overlay,
}

impl Gpu {
    async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
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

        let seed = random_cells(GRID_WIDTH * GRID_HEIGHT, 0x1234_5678);
        let cell_buffers = [0, 1].map(|i| {
            create_buffer_init(
                &device,
                &format!("cells {i}"),
                bytemuck::cast_slice(&seed),
                wgpu::BufferUsages::STORAGE,
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
                entries: &[bind(0, &params_buffer), bind(1, &cell_buffers[i])],
            })
        });

        let mut overlay = overlay::Overlay::new(&device, config.format);
        overlay.resize(config.width, config.height, window.scale_factor());

        Self {
            overlay,
            window,
            surface,
            device,
            queue,
            config,
            compute_pipeline,
            render_pipeline,
            compute_bind_groups,
            render_bind_groups,
            current: 0,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.overlay.resize(width, height, self.window.scale_factor());
        }
    }

    /// Advance one generation and draw it.
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
        let mut encoder = self.device.create_command_encoder(&Default::default());

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.compute_bind_groups[self.current], &[]);
            pass.dispatch_workgroups(
                GRID_WIDTH.div_ceil(WORKGROUP_SIZE),
                GRID_HEIGHT.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }
        self.current = 1 - self.current;

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

            self.overlay.tick();
            self.overlay.draw(&self.queue, &mut pass);
        }

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.queue.present(frame);
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
        let attrs = Window::default_attributes()
            .with_title("GPU Conway's Game of Life")
            .with_inner_size(winit::dpi::LogicalSize::new(768, 768));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.gpu = Some(pollster::block_on(Gpu::new(window)));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => gpu.resize(size.width, size.height),
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
