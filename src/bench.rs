//! Headless benchmark: `cargo run --release -- --bench`.
//!
//! Runs each engine flat out (no window, no vsync, no frame budget) on the same
//! starting worlds and reports generations/second. Timing includes waiting for the
//! GPU to finish, and for the sparse engine its CPU bookkeeping.

use std::time::{Duration, Instant};

use crate::dense::DenseEngine;
use crate::engine::{Engine, GpuContext, human};
use crate::patterns;
use crate::sparse::SparseEngine;

/// How long each measurement runs (after warm-up).
const MEASURE: Duration = Duration::from_secs(3);
const WARMUP_GENERATIONS: u32 = 64;
const BATCH: u32 = 64;

struct Result {
    scenario: String,
    engine: &'static str,
    gens_per_sec: f64,
}

/// Generations per second, and the wall time measured.
fn time(ctx: &GpuContext, engine: &mut dyn Engine) -> (f64, Duration) {
    let wait = || {
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    };
    engine.step(WARMUP_GENERATIONS);
    wait();
    let start = Instant::now();
    let mut generations = 0u64;
    while start.elapsed() < MEASURE {
        engine.step(BATCH);
        generations += u64::from(BATCH);
        // Sync now and then so we measure completed work, not queued work.
        if generations % u64::from(BATCH * 4) == 0 {
            wait();
        }
    }
    wait();
    let elapsed = start.elapsed();
    (generations as f64 / elapsed.as_secs_f64(), elapsed)
}

pub fn run(ctx: &GpuContext) {
    let mut results: Vec<Result> = Vec::new();
    let mut record = |scenario: &str, engine: &'static str, setup: String, gps: f64| {
        println!("  {scenario:<34} {engine:<7} {setup:<28} {gps:>10.0} gens/s");
        results.push(Result { scenario: scenario.to_owned(), engine, gens_per_sec: gps });
    };

    println!("Benchmarking ({}s per measurement)...", MEASURE.as_secs());

    // 1. Small patterns: dense at its default size vs sparse.
    for (name, pattern) in [
        ("acorn", patterns::ACORN),
        ("bunnies", patterns::BUNNIES),
        ("glider gun", patterns::GOSPER_GLIDER_GUN),
    ] {
        let cells = patterns::centered(pattern);
        let scenario = format!("{name} (small pattern)");
        let mut dense = pollster::block_on(DenseEngine::new(ctx));
        dense.load(&cells);
        let setup = format!("grid {}", dense.info());
        let (gps, _) = time(ctx, &mut dense);
        record(&scenario, "dense", setup, gps);
        drop(dense);

        let mut sparse = SparseEngine::new(ctx);
        sparse.load(&cells);
        let (gps, _) = time(ctx, &mut sparse);
        // A correctness check too: acorn ends at 633 cells, bunnies at 1744.
        let pop = sparse.population().map_or(0, |(p, _)| p);
        record(&scenario, "sparse", format!("{} tiles, pop {pop}", sparse.tile_count()), gps);
    }

    // 2. Random soup filling an S x S square, with the dense grid sized to fit it
    //    exactly: the fairest fight for the dense engine.
    for side in [256u32, 1024, 2048, 4096, 8192, 16384] {
        let scenario = format!("soup {side}x{side}, grid = soup size");
        let mut dense = pollster::block_on(DenseEngine::with_side(ctx, side));
        dense.random(); // fills the whole grid
        let setup = dense.info();
        let (gps, _) = time(ctx, &mut dense);
        record(&scenario, "dense", setup, gps);
        drop(dense);

        let mut sparse = SparseEngine::new(ctx);
        sparse.load(&patterns::random_square(side as i32, crate::engine::time_seed()));
        let start_tiles = sparse.tile_count();
        sparse.take_bookkeeping_time();
        let (gps, wall) = time(ctx, &mut sparse);
        // Share of wall time the CPU spent on bookkeeping (it overlaps GPU work).
        let cpu = 100.0 * sparse.take_bookkeeping_time().as_secs_f64() / wall.as_secs_f64();
        let setup = format!("{start_tiles} -> {} tiles, cpu {cpu:.0}%", sparse.tile_count());
        record(&scenario, "sparse", setup, gps);
    }

    // Summary: winner per scenario.
    println!("\nScenario                            winner  ratio");
    let mut i = 0;
    while i + 1 < results.len() {
        let (a, b) = (&results[i], &results[i + 1]);
        let (win, lose) = if a.gens_per_sec >= b.gens_per_sec { (a, b) } else { (b, a) };
        println!(
            "{:<35} {:<7} {:.1}x  ({} vs {} gens/s)",
            a.scenario,
            win.engine,
            win.gens_per_sec / lose.gens_per_sec.max(1e-9),
            human(win.gens_per_sec, &[(1e6, "M"), (1e3, "K")]),
            human(lose.gens_per_sec, &[(1e6, "M"), (1e3, "K")]),
        );
        i += 2;
    }
}
