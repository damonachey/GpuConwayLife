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

/// An empty `width` x `height` grid with `pattern` stamped in the centre.
pub fn centered(pattern: &[&str], width: u32, height: u32) -> Vec<u32> {
    let (w, h) = (width as usize, height as usize);
    let pat_h = pattern.len();
    let pat_w = pattern.iter().map(|r| r.len()).max().unwrap_or(0);
    let x0 = w.saturating_sub(pat_w) / 2;
    let y0 = h.saturating_sub(pat_h) / 2;

    let mut cells = vec![0; w * h];
    for (dy, row) in pattern.iter().enumerate() {
        for (dx, b) in row.bytes().enumerate() {
            let (x, y) = (x0 + dx, y0 + dy);
            if b == b'O' && x < w && y < h {
                cells[y * w + x] = 1;
            }
        }
    }
    cells
}
