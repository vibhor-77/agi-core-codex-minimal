## Manual ARC Notes

I manually inspected these ARC-AGI-1 training tasks:

- `67a3c6ac`: horizontal mirror
- `68b16354`: vertical mirror
- `74dd1130`: transpose
- `3c9b0459`: 180-degree rotation
- `ed36ccf7`: 90-degree rotation-like remap
- `9dfd6313`: copy lower-triangular content across the diagonal
- `1cf80156`: crop to the support box
- `28bf18c6`: crop support, then duplicate horizontally
- `6d0aefbc`: concatenate an object with its mirrored copy
- `25ff71a9`: project occupied cells one row down
- `794b24be`: project occupied columns to a canonical top row
- `a1570a43`: move an object relative to corner markers
- `5521c0d9`: re-layout separate rectangles into canonical slots
- `d4f3cd78`: fill an enclosure interior and extend a line
- `b27ca6d3`: draw a local frame around a sparse configuration

## Distilled Basics

The easiest ARC tasks do not really require named transforms like `mirror` or `transpose`.
They can be described more primitively as coordinate remaps over:

- cell value
- row, col
- height, width

Examples:

- horizontal mirror: `(r, c) -> (r, width - 1 - c)`
- vertical mirror: `(r, c) -> (height - 1 - r, c)`
- transpose: `(r, c) -> (c, r)`
- 180-degree rotation: `(r, c) -> (height - 1 - r, width - 1 - c)`

That suggests a better substrate than naming those operations directly.

## Primitive Substrate To Aim For

- values: `int`, `bool`
- structure: `grid`, `mask`, `coord`, `box`
- primitive operations:
  - equality / nonzero tests
  - boolean mask operations
  - coordinate arithmetic
  - read cell / write cell
  - copy / compose

The current code mirrors that direction with `IntExpr`, `MaskExpr`, and `GridExpr`.

Higher-level operations should ideally be discovered:

- mirror / transpose / rotate from coordinate remaps
- diagonals / borders / corners from coordinate predicates
- crop from `bbox(mask)`
- local rewrite from `mask -> select -> write back`
- repetition / projection / packing from repeated use of the same low-level operations

Many tasks also require value-level distinctions that `nonzero` alone cannot express.
Examples:

- `794b24be`: the occupied cell projects upward but changes from `1` to `2`
- `d4f3cd78`: an enclosure interior is filled with `8` rather than copied from an existing nonzero cell
- `b27ca6d3`: a sparse configuration of `2` cells induces a new frame of `3`

That suggests the substrate also needs low-level cell-symbol access:

- read cell value at a coordinate
- compare a cell value to a symbol
- paint a symbol back onto a grid

These are still primitive in the same sense as characters or integers in a programming language.
They let color-specific behavior emerge without seeding named ARC tactics.

## Why This Matters

If the substrate is too high-level, progress can come from hand-seeded human concepts.
If the substrate is low-level but expressive, compounding can come from learned reusable programs
instead of from adding new named tactics.
