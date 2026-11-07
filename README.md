# GpuConwayLife

Conway's Game of Life running entirely on the GPU, written in Rust with
[wgpu](https://wgpu.rs) and [winit](https://github.com/rust-windowing/winit).

## How it works

- Cells are **bit-packed**, 32 to a `u32`, in two GPU storage buffers
  (ping-pong). The default grid is **16,384 × 16,384 ≈ 268 million cells**
  (32 MiB per buffer). `LIFE_GRID` can go up to what the GPU allows, up to
  2 GiB per buffer: on an RTX 3070 that's 131,040 × 131,040 ≈ 17.2 billion
  cells. If the allocation fails, the grid shrinks until it fits.
- **Edges are dead**: everything beyond the grid counts as empty, so patterns
  that reach the border die there instead of wrapping.
- A **compute shader** (`src/life.wgsl`) computes 32 cells per thread: it shifts
  the 9 surrounding words into 8 neighbour words and adds them with a bitwise
  adder, so all 32 neighbour counts happen in parallel.
- Random soup is generated **on the GPU** (a hash per word), and the
  population is counted on the GPU and read back in the background.
- A **fragment shader** (`src/render.wgsl`) draws the middle of the grid, one
  cell per screen pixel. Cell data never round-trips through the CPU.
- A small overlay (`src/overlay.wgsl`, built-in 3x5 pixel font) shows FPS,
  generations/second, cell updates/second, population, speed and grid size.
- Each frame does at most ~100 ms of simulation, so huge grids at high speed
  stay responsive (excess requested generations are dropped).

Measured on an RTX 3070: ~1.4 trillion cell updates/second on large grids,
versus ~22 billion with the previous one-`u32`-per-cell version. Verified with
the acorn, which reaches its known final population of 633.

Set `LIFE_GRID=<side>` to pick another square grid size, e.g. `LIFE_GRID=4096`
or `LIFE_GRID=131072` (capped at what the GPU allows).

## Controls

| Key     | Action                                  |
| ------- | --------------------------------------- |
| Space   | Pause / resume                          |
| N       | Step one generation (while paused; hold to repeat) |
| R       | Reseed with random cells                |
| 1       | Clear and place a Gosper glider gun in the centre |
| 2       | Clear and place an acorn in the centre  |
| 3       | Clear and place Bunnies in the centre   |
| 4       | Clear and place Ark1 in the centre      |
| + / -   | Double / halve speed (1/16 to 1024 generations per frame) |

## Run

```sh
cargo run --release
```

On Windows this needs the MSVC C++ build tools (the "Desktop development with C++"
workload in Visual Studio or Build Tools).
