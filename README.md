# GpuConwayLife

Conway's Game of Life running entirely on the GPU, written in Rust with
[wgpu](https://wgpu.rs) and [winit](https://github.com/rust-windowing/winit).

## How it works

- The grid (1000×1000, wrapping edges, one cell per screen pixel) is stored in
  two GPU storage buffers.
- Each frame, a **compute shader** (`src/life.wgsl`) reads one buffer and writes
  the next generation into the other, and the buffers swap roles (ping-pong).
- A **fragment shader** (`src/render.wgsl`) draws the newest buffer straight to
  the window. Cell data never round-trips through the CPU after the initial seed.

- A small overlay (`src/overlay.wgsl`, built-in 3x5 pixel font) shows FPS,
  generation count, generations/second and speed.

## Controls

| Key     | Action                                  |
| ------- | --------------------------------------- |
| Space   | Pause / resume                          |
| N       | Step one generation (while paused; hold to repeat) |
| R       | Reseed with random cells                |
| 1       | Clear and place a Gosper glider gun in the centre |
| + / -   | Double / halve speed (1/16 to 1024 generations per frame) |

## Run

```sh
cargo run --release
```

On Windows this needs the MSVC C++ build tools (the "Desktop development with C++"
workload in Visual Studio or Build Tools).
