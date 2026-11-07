# GpuConwayLife

Conway's Game of Life running entirely on the GPU, written in Rust with
[wgpu](https://wgpu.rs) and [winit](https://github.com/rust-windowing/winit).

## How it works

- The grid (256×256, wrapping edges) is stored in two GPU storage buffers.
- Each frame, a **compute shader** (`src/life.wgsl`) reads one buffer and writes
  the next generation into the other, and the buffers swap roles (ping-pong).
- A **fragment shader** (`src/render.wgsl`) draws the newest buffer straight to
  the window. Cell data never round-trips through the CPU after the initial seed.

## Run

```sh
cargo run --release
```

On Windows this needs the MSVC C++ build tools (the "Desktop development with C++"
workload in Visual Studio or Build Tools).
