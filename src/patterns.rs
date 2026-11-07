//! Built-in starting patterns, drawn as ASCII art (`O` = alive, anything else = dead).

/// Bill Gosper's glider gun (1970): the first known pattern that grows forever,
/// firing a new glider down and to the right every 30 generations.
pub const GOSPER_GLIDER_GUN: &[&str] = &[
    "........................O...........",
    "......................O.O...........",
    "............OO......OO............OO",
    "...........O...O....OO............OO",
    "OO........O.....O...OO..............",
    "OO........O...O.OO....O.O...........",
    "..........O.....O.......O...........",
    "...........O...O....................",
    "............OO......................",
];

/// The acorn (as used by MinimalConwayLife): 7 cells that take 5206 generations
/// to stabilise, throwing off 13 gliders along the way.
pub const ACORN: &[&str] = &[
    ".O.....",
    "...O...",
    "OO..OOO",
];

/// Bunnies (Robert Wainwright and Andrew Trevorrow): 9 cells that take 17332
/// generations to stabilise; the parent of the "rabbits" family of methuselahs.
pub const BUNNIES: &[&str] = &[
    "O.....O.",
    "..O...O.",
    "..O..O.O",
    ".O.O....",
];

/// Ark1 (Nick Gotts): 16 cells in two separated pieces that take 736692
/// generations to stabilise. From Golly's pattern collection.
pub const ARK1: &[&str] = &[
    "...........................O....",
    "............................O...",
    ".............................O..",
    "............................O...",
    "...........................O....",
    ".............................OOO",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "OO..............................",
    "..O.............................",
    "..O.............................",
    "...OOOO.........................",
];

/// Live cells of `pattern` as (x, y) offsets centred on (0, 0).
pub fn centered(pattern: &[&str]) -> Vec<(i32, i32)> {
    let h = pattern.len() as i32;
    let w = pattern.iter().map(|r| r.len()).max().unwrap_or(0) as i32;
    let mut cells = Vec::new();
    for (y, row) in pattern.iter().enumerate() {
        for (x, b) in row.bytes().enumerate() {
            if b == b'O' {
                cells.push((x as i32 - w / 2, y as i32 - h / 2));
            }
        }
    }
    cells
}

/// A `size` x `size` square of random soup (about 25% alive) centred on the origin.
pub fn random_square(size: i32, mut state: u32) -> Vec<(i32, i32)> {
    let mut cells = Vec::new();
    for y in 0..size {
        for x in 0..size {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if state % 4 == 0 {
                cells.push((x - size / 2, y - size / 2));
            }
        }
    }
    cells
}