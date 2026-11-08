// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <span>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/div_ceil.h"
#include "video_core/gpu.h"
#include "video_core/textures/decoders.h"

namespace Tegra::Texture {
namespace {
constexpr u32 NativeGobOffset(u32 offset, bool native_tiling) {
    if (!native_tiling) return offset;
    const u32 difference = ((offset >> 5) ^ (offset >> 6)) & 1;
    return offset ^ (difference << 5) ^ (difference << 6);
}
template <u32 mask>
constexpr u32 pdep(u32 value) {
    u32 result = 0;
    u32 m = mask;
    for (u32 bit = 1; m; bit += bit) {
        if (value & bit)
            result |= m & (~m + 1);
        m &= m - 1;
    }
    return result;
}

template <u32 mask, u32 incr_amount>
void incrpdep(u32& value) {
    static constexpr u32 swizzled_incr = pdep<mask>(incr_amount);
    value = ((value | ~mask) + swizzled_incr) & mask;
}

template <bool TO_LINEAR, u32 BYTES_PER_PIXEL>
void SwizzleImpl(std::span<u8> output, std::span<const u8> input, u32 width, u32 height, u32 depth,
                 u32 block_height, u32 block_depth, u32 stride, bool native_tiling) {
    // The origin of the transformation can be configured here, leave it as zero as the current API
    // doesn't expose it.
    static constexpr u32 origin_x = 0;
    static constexpr u32 origin_y = 0;
    static constexpr u32 origin_z = 0;

    // We can configure here a custom pitch
    // As it's not exposed 'width * BYTES_PER_PIXEL' will be the expected pitch.
    const u32 pitch = width * BYTES_PER_PIXEL;

    const u32 gobs_in_x = Common::DivCeilLog2(stride, GOB_SIZE_X_SHIFT);
    const u32 block_size = gobs_in_x << (GOB_SIZE_SHIFT + block_height + block_depth);
    const u32 slice_size =
        Common::DivCeilLog2(height, block_height + GOB_SIZE_Y_SHIFT) * block_size;

    const u32 block_height_mask = (1U << block_height) - 1;
    const u32 block_depth_mask = (1U << block_depth) - 1;
    const u32 x_shift = GOB_SIZE_SHIFT + block_height + block_depth;

    for (u32 slice = 0; slice < depth; ++slice) {
        const u32 z = slice + origin_z;
        const u32 offset_z = (z >> block_depth) * slice_size +
                             ((z & block_depth_mask) << (GOB_SIZE_SHIFT + block_height));
        for (u32 line = 0; line < height; ++line) {
            const u32 y = line + origin_y;
            const u32 swizzled_y = pdep<SWIZZLE_Y_BITS>(y);

            const u32 block_y = y >> GOB_SIZE_Y_SHIFT;
            const u32 offset_y = (block_y >> block_height) * block_size +
                                 ((block_y & block_height_mask) << GOB_SIZE_SHIFT);

            u32 swizzled_x = pdep<SWIZZLE_X_BITS>(origin_x * BYTES_PER_PIXEL);
            for (u32 column = 0; column < width;
                 ++column, incrpdep<SWIZZLE_X_BITS, BYTES_PER_PIXEL>(swizzled_x)) {
                const u32 x = (column + origin_x) * BYTES_PER_PIXEL;
                const u32 offset_x = (x >> GOB_SIZE_X_SHIFT) << x_shift;

                const u32 base_swizzled_offset = offset_z + offset_y + offset_x;
                const u32 swizzled_offset = NativeGobOffset(base_swizzled_offset + (swizzled_x | swizzled_y), native_tiling);

                const u32 unswizzled_offset =
                    slice * pitch * height + line * pitch + column * BYTES_PER_PIXEL;

                u8* const dst = &output[TO_LINEAR ? swizzled_offset : unswizzled_offset];
                const u8* const src = &input[TO_LINEAR ? unswizzled_offset : swizzled_offset];

                std::memcpy(dst, src, BYTES_PER_PIXEL);
            }
        }
    }
}

template <bool TO_LINEAR, u32 BYTES_PER_PIXEL>
void SwizzleSubrectImpl(std::span<u8> output, std::span<const u8> input, u32 width, u32 height,
                        u32 depth, u32 origin_x, u32 origin_y, u32 extent_x, u32 num_lines,
                        u32 block_height, u32 block_depth, u32 pitch_linear, bool native_tiling) {
    // The origin of the transformation can be configured here, leave it as zero as the current API
    // doesn't expose it.
    static constexpr u32 origin_z = 0;

    // We can configure here a custom pitch
    // As it's not exposed 'width * BYTES_PER_PIXEL' will be the expected pitch.
    const u32 pitch = pitch_linear;
    const u32 stride = Common::AlignUpLog2(width * BYTES_PER_PIXEL, GOB_SIZE_X_SHIFT);

    const u32 gobs_in_x = Common::DivCeilLog2(stride, GOB_SIZE_X_SHIFT);
    const u32 block_size = gobs_in_x << (GOB_SIZE_SHIFT + block_height + block_depth);
    const u32 slice_size =
        Common::DivCeilLog2(height, block_height + GOB_SIZE_Y_SHIFT) * block_size;

    const u32 block_height_mask = (1U << block_height) - 1;
    const u32 block_depth_mask = (1U << block_depth) - 1;
    const u32 x_shift = GOB_SIZE_SHIFT + block_height + block_depth;

    u32 unprocessed_lines = num_lines;
    u32 extent_y = (std::min)(num_lines, height - origin_y);

    for (u32 slice = 0; slice < depth; ++slice) {
        const u32 z = slice + origin_z;
        const u32 offset_z = (z >> block_depth) * slice_size +
                             ((z & block_depth_mask) << (GOB_SIZE_SHIFT + block_height));
        const u32 lines_in_y = (std::min)(unprocessed_lines, extent_y);
        for (u32 line = 0; line < lines_in_y; ++line) {
            const u32 y = line + origin_y;
            const u32 swizzled_y = pdep<SWIZZLE_Y_BITS>(y);

            const u32 block_y = y >> GOB_SIZE_Y_SHIFT;
            const u32 offset_y = (block_y >> block_height) * block_size +
                                 ((block_y & block_height_mask) << GOB_SIZE_SHIFT);

            u32 swizzled_x = pdep<SWIZZLE_X_BITS>(origin_x * BYTES_PER_PIXEL);
            for (u32 column = 0; column < extent_x;
                 ++column, incrpdep<SWIZZLE_X_BITS, BYTES_PER_PIXEL>(swizzled_x)) {
                const u32 x = (column + origin_x) * BYTES_PER_PIXEL;
                const u32 offset_x = (x >> GOB_SIZE_X_SHIFT) << x_shift;

                const u32 base_swizzled_offset = offset_z + offset_y + offset_x;
                const u32 swizzled_offset = NativeGobOffset(base_swizzled_offset + (swizzled_x | swizzled_y), native_tiling);

                const u32 unswizzled_offset =
                    slice * pitch * height + line * pitch + column * BYTES_PER_PIXEL;

                u8* const dst = &output[TO_LINEAR ? swizzled_offset : unswizzled_offset];
                const u8* const src = &input[TO_LINEAR ? unswizzled_offset : swizzled_offset];

                std::memcpy(dst, src, BYTES_PER_PIXEL);
            }
        }
        unprocessed_lines -= lines_in_y;
        if (unprocessed_lines == 0) {
            return;
        }
    }
}

template <bool TO_LINEAR>
void Swizzle(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel, u32 width,
             u32 height, u32 depth, u32 block_height, u32 block_depth, u32 stride_alignment, bool native_tiling) {
    switch (bytes_per_pixel) {
#define BPP_CASE(x)                                                                                \
    case x:                                                                                        \
        return SwizzleImpl<TO_LINEAR, x>(output, input, width, height, depth, block_height,        \
                                         block_depth, stride_alignment, native_tiling);
        BPP_CASE(1)
        BPP_CASE(2)
        BPP_CASE(3)
        BPP_CASE(4)
        BPP_CASE(6)
        BPP_CASE(8)
        BPP_CASE(12)
        BPP_CASE(16)
#undef BPP_CASE
    default:
        ASSERT_MSG(false, "Invalid bytes_per_pixel={}", bytes_per_pixel);
        break;
    }
}

// EDEN_NX2_FAST_UNSWIZZLE: block linear to linear for single-slice images whose rows are a
// multiple of 16 bytes, a GOB at a time. A GOB (64 bytes x 8 rows) is 32 runs of 16 contiguous
// bytes whose places in it (native tiling included, it only swaps runs) come from a table, so
// the per-16-byte address arithmetic of SwizzleImpl becomes a lookup.
struct GobRunTable {
    std::array<u16, 32> offsets; ///< Of the run at row r, column c (16 bytes) at index r * 4 + c
};

constexpr GobRunTable MakeGobRunTable(bool native_tiling) {
    GobRunTable table{};
    for (u32 row = 0; row < GOB_SIZE_Y; ++row) {
        for (u32 column = 0; column < 4; ++column) {
            const u32 offset = pdep<SWIZZLE_X_BITS>(column * 16) | pdep<SWIZZLE_Y_BITS>(row);
            table.offsets[row * 4 + column] = static_cast<u16>(NativeGobOffset(offset, native_tiling));
        }
    }
    return table;
}

constexpr GobRunTable plain_gob_runs = MakeGobRunTable(false);
constexpr GobRunTable native_gob_runs = MakeGobRunTable(true);

void UnswizzleGobs(std::span<u8> output, std::span<const u8> input, u32 pitch, u32 height,
                   u32 block_height, u32 stride, bool native_tiling) {
    const GobRunTable& table = native_tiling ? native_gob_runs : plain_gob_runs;
    const u32 gobs_in_x = Common::DivCeilLog2(stride, GOB_SIZE_X_SHIFT);
    const u32 block_size = gobs_in_x << (GOB_SIZE_SHIFT + block_height);
    const u32 block_height_mask = (1U << block_height) - 1;
    const u32 x_shift = GOB_SIZE_SHIFT + block_height;
    const u32 full_gobs = pitch / GOB_SIZE_X;
    const u32 tail_runs = (pitch % GOB_SIZE_X) / 16;
    const u8* const in = input.data();
    u8* const out = output.data();
    const u32 gob_rows = Common::DivCeilLog2(height, GOB_SIZE_Y_SHIFT);
    for (u32 gob_y = 0; gob_y < gob_rows; ++gob_y) {
        const u32 offset_y = (gob_y >> block_height) * block_size +
                             ((gob_y & block_height_mask) << GOB_SIZE_SHIFT);
        const u32 first_line = gob_y * GOB_SIZE_Y;
        const u32 lines = (std::min)(GOB_SIZE_Y, height - first_line);
        for (u32 gob_x = 0; gob_x <= full_gobs; ++gob_x) {
            const u32 runs = gob_x < full_gobs ? 4 : tail_runs;
            if (runs == 0) {
                break;
            }
            const u8* const gob = in + offset_y + (gob_x << x_shift);
            u8* const row_base = out + size_t{first_line} * pitch + gob_x * GOB_SIZE_X;
            for (u32 line = 0; line < lines; ++line) {
                u8* const dst = row_base + size_t{line} * pitch;
                const u16* const run = &table.offsets[line * 4];
                for (u32 column = 0; column < runs; ++column) {
                    std::memcpy(dst + column * 16, gob + run[column], 16);
                }
            }
        }
    }
}

bool FastUnswizzleEnabled() {
    static const bool enabled = std::getenv("EDEN_NX2_FAST_UNSWIZZLE") != nullptr;
    return enabled;
}

} // Anonymous namespace

void UnswizzleTexture(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel,
                      u32 width, u32 height, u32 depth, u32 block_height, u32 block_depth,
                      u32 stride_alignment, bool native_tiling) {
    const u32 stride = Common::AlignUpLog2(width, stride_alignment) * bytes_per_pixel;
    if (depth == 1 && block_depth == 0 && ((width * bytes_per_pixel) & 15) == 0 &&
        FastUnswizzleEnabled()) {
        UnswizzleGobs(output, input, width * bytes_per_pixel, height, block_height, stride,
                      native_tiling);
        return;
    }
    const u32 new_bpp = (std::min)(4U, static_cast<u32>(std::countr_zero(width * bytes_per_pixel)));
    width = (width * bytes_per_pixel) >> new_bpp;
    bytes_per_pixel = 1U << new_bpp;
    Swizzle<false>(output, input, bytes_per_pixel, width, height, depth, block_height, block_depth,
                   stride, native_tiling);
}

void SwizzleTexture(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel, u32 width,
                    u32 height, u32 depth, u32 block_height, u32 block_depth,
                    u32 stride_alignment, bool native_tiling) {
    const u32 stride = Common::AlignUpLog2(width, stride_alignment) * bytes_per_pixel;
    const u32 new_bpp = (std::min)(4U, static_cast<u32>(std::countr_zero(width * bytes_per_pixel)));
    width = (width * bytes_per_pixel) >> new_bpp;
    bytes_per_pixel = 1U << new_bpp;
    Swizzle<true>(output, input, bytes_per_pixel, width, height, depth, block_height, block_depth,
                  stride, native_tiling);
}

void SwizzleSubrect(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel, u32 width,
                    u32 height, u32 depth, u32 origin_x, u32 origin_y, u32 extent_x, u32 extent_y,
                    u32 block_height, u32 block_depth, u32 pitch_linear, bool native_tiling) {
    switch (bytes_per_pixel) {
#define BPP_CASE(x)                                                                                \
    case x:                                                                                        \
        return SwizzleSubrectImpl<true, x>(output, input, width, height, depth, origin_x,          \
                                           origin_y, extent_x, extent_y, block_height,             \
                                           block_depth, pitch_linear, native_tiling);
        BPP_CASE(1)
        BPP_CASE(2)
        BPP_CASE(3)
        BPP_CASE(4)
        BPP_CASE(6)
        BPP_CASE(8)
        BPP_CASE(12)
        BPP_CASE(16)
#undef BPP_CASE
    default:
        ASSERT_MSG(false, "Invalid bytes_per_pixel={}", bytes_per_pixel);
        break;
    }
}

void UnswizzleSubrect(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel,
                      u32 width, u32 height, u32 depth, u32 origin_x, u32 origin_y, u32 extent_x,
                      u32 extent_y, u32 block_height, u32 block_depth, u32 pitch_linear, bool native_tiling) {
    switch (bytes_per_pixel) {
#define BPP_CASE(x)                                                                                \
    case x:                                                                                        \
        return SwizzleSubrectImpl<false, x>(output, input, width, height, depth, origin_x,         \
                                            origin_y, extent_x, extent_y, block_height,            \
                                            block_depth, pitch_linear, native_tiling);
        BPP_CASE(1)
        BPP_CASE(2)
        BPP_CASE(3)
        BPP_CASE(4)
        BPP_CASE(6)
        BPP_CASE(8)
        BPP_CASE(12)
        BPP_CASE(16)
#undef BPP_CASE
    default:
        ASSERT_MSG(false, "Invalid bytes_per_pixel={}", bytes_per_pixel);
        break;
    }
}

std::size_t CalculateSize(bool tiled, u32 bytes_per_pixel, u32 width, u32 height, u32 depth,
                          u32 block_height, u32 block_depth) {
    if (tiled) {
        const u32 aligned_width = Common::AlignUpLog2(width * bytes_per_pixel, GOB_SIZE_X_SHIFT);
        const u32 aligned_height = Common::AlignUpLog2(height, GOB_SIZE_Y_SHIFT + block_height);
        const u32 aligned_depth = Common::AlignUpLog2(depth, GOB_SIZE_Z_SHIFT + block_depth);
        return aligned_width * aligned_height * aligned_depth;
    } else {
        return width * height * depth * bytes_per_pixel;
    }
}

u64 GetGOBOffset(u32 width, u32 height, u32 dst_x, u32 dst_y, u32 block_height,
                 u32 bytes_per_pixel) {
    auto div_ceil = [](const u32 x, const u32 y) { return ((x + y - 1) / y); };
    const u32 gobs_in_block = 1 << block_height;
    const u32 y_blocks = GOB_SIZE_Y << block_height;
    const u32 x_per_gob = GOB_SIZE_X / bytes_per_pixel;
    const u32 x_blocks = div_ceil(width, x_per_gob);
    const u32 block_size = GOB_SIZE * gobs_in_block;
    const u32 stride = block_size * x_blocks;
    const u32 base = (dst_y / y_blocks) * stride + (dst_x / x_per_gob) * block_size;
    const u32 relative_y = dst_y % y_blocks;
    return base + (relative_y / GOB_SIZE_Y) * GOB_SIZE;
}

} // namespace Tegra::Texture
