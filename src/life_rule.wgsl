// Conway's rule for 32 cells at once, shared by both engines' compute shaders
// (Rust prepends this file to them; see `engine::shader`).
//
// Cells are bit-packed: bit i of a word is the cell at x = wx * 32 + i. Given the
// word `m` and its eight neighbouring words (row above: ul u ur, same row: ml mr,
// row below: dl d dr), returns the next generation of `m`'s 32 cells.

// Per-bit counter: s.x = ones, s.y = twos, s.z = "four or more" (sticky).
// That's all Life needs: alive next iff count == 3, or count == 2 and alive now.
fn add_bit(s: ptr<function, vec3<u32>>, x: u32) {
    let c0 = (*s).x & x;
    (*s).x ^= x;
    let c1 = (*s).y & c0;
    (*s).y ^= c0;
    (*s).z |= c1;
}

fn next_word(ul: u32, u: u32, ur: u32, ml: u32, m: u32, mr: u32, dl: u32, d: u32, dr: u32) -> u32 {
    // West neighbour of bit i is bit i-1 (from the left word at bit 0);
    // east neighbour is bit i+1 (from the right word at bit 31).
    var s = vec3<u32>(0u);
    add_bit(&s, (u << 1u) | (ul >> 31u));
    add_bit(&s, u);
    add_bit(&s, (u >> 1u) | (ur << 31u));
    add_bit(&s, (m << 1u) | (ml >> 31u));
    add_bit(&s, (m >> 1u) | (mr << 31u));
    add_bit(&s, (d << 1u) | (dl >> 31u));
    add_bit(&s, d);
    add_bit(&s, (d >> 1u) | (dr << 31u));
    // count 2 -> (ones=0, twos=1), count 3 -> (ones=1, twos=1), and fewer than 4.
    return s.y & ~s.z & (s.x | m);
}
