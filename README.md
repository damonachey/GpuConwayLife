# GpuConwayLife

Conway's Game of Life running entirely on the GPU, written in Rust with
[wgpu](https://wgpu.rs) and [winit](https://github.com/rust-windowing/winit).

Two simulation engines share one app; press **E** to switch between them.
Switching frees the current engine's GPU memory, builds the other one and
restarts whatever was last loaded (random soup or a pattern).

## Dense engine (`src/dense.rs`) — the default

- Cells are **bit-packed**, 32 to a `u32`, in two GPU storage buffers
  (ping-pong). The default grid is **16,384 × 16,384 ≈ 268 million cells**
  (32 MiB per buffer). `LIFE_GRID` can go up to what the GPU allows, up to
  2 GiB per buffer: on an RTX 3070 that's 131,040 × 131,040 ≈ 17.2 billion
  cells. If the allocation fails, the grid shrinks until it fits.
- **Edges are dead**: everything beyond the grid counts as empty, so patterns
  that reach the border die there instead of wrapping.
- The compute shader (`src/dense_life.wgsl`) computes 32 cells per thread: it
  shifts the 9 surrounding words into 8 neighbour words and adds them with a
  bitwise adder, so all 32 neighbour counts happen in parallel.
- Random soup fills the whole grid and is generated **on the GPU**; the
  population is counted on the GPU and read back in the background.
- Measured on an RTX 3070: ~1.4 trillion cell updates/second, versus ~22 billion
  with the earlier one-`u32`-per-cell version.

## Sparse engine (`src/sparse.rs`, `src/tiles.rs`)

- The world is **unbounded**: it's cut into 64×64 tiles, and only tiles near
  live cells exist. Tiles are **bit-packed** (64 rows × 2 words = 512 bytes), so
  the GPU pool holds up to **1,048,576 tiles ≈ 4.3 billion cells** (2 × 512 MiB).
- The compute shader (`src/sparse_life.wgsl`) steps one tile per 128-thread
  workgroup, one thread per word, with the same bitwise adder as the dense
  engine. Words beyond a tile's edge come from neighbour tiles via a per-tile
  neighbour table. It also reports per tile whether it's alive, which neighbours
  it needs, and its population.
- Life spreads at most one cell per generation, so the CPU only looks every
  **16 generations**: it reads back 4 bytes per tile, allocates neighbours
  wherever life is within 32 cells of an edge, and frees empty tiles. Readbacks
  are **pipelined** (applied one pass late, from double-buffered staging), so
  the GPU never waits on the CPU; the 32-cell margin covers the extra delay.
- Random soup is a 1000×1000 square at the origin. **T** shades the allocated
  tiles; the overlay shows how full the tile pool is.

Both engines draw one cell per screen pixel, centred on the world's middle, and
both are verified with the acorn, which reaches its known final population of
633 (the dense engine until its escaping gliders reach the dead edges).

Each frame does at most ~100 ms of simulation, so big worlds at high speed stay
responsive (excess requested generations are dropped).

## Controls

| Key     | Action                                  |
| ------- | --------------------------------------- |
| Space   | Pause / resume                          |
| N       | Step one generation (while paused; hold to repeat) |
| R       | Random soup                             |
| 1       | Gosper glider gun                       |
| 2       | Acorn                                   |
| 3       | Bunnies                                 |
| 4       | Ark1                                    |
| E       | Switch engine (dense ↔ sparse), restarting the current pattern |
| T       | Show / hide allocated tiles (sparse engine) |
| + / -   | Double / halve speed (1/16 to 1024 generations per frame) |

## Run

```sh
cargo run --release
```

Set `LIFE_GRID=<side>` to pick the dense engine's grid size, e.g.
`LIFE_GRID=4096` or `LIFE_GRID=131072` (capped at what the GPU allows).

On Windows this needs the MSVC C++ build tools (the "Desktop development with C++"
workload in Visual Studio or Build Tools).
