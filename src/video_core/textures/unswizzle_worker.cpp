// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <latch>
#include <vector>

#include "common/alignment.h"
#include "common/div_ceil.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "video_core/textures/decoders.h"
#include "video_core/textures/unswizzle_worker.h"
#include "video_core/textures/workers.h"

namespace Tegra::Texture {

namespace {
// EDEN_NX2_FAST_UNSWIZZLE: images from 512 KiB (the default path waits for 2 MiB and rows
// that are not a multiple of 4 bytes) are split into up to 4 bands of block rows, each
// unswizzled on its own thread (this one takes the last), straight from the input: Ocarina of
// Time's ~1.3 MB textures were unswizzled on the GPU back thread alone (~7% of it).
bool TryUnswizzleBands(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel,
                       u32 width, u32 height, u32 depth, u32 block_height, u32 block_depth,
                       u32 stride_alignment, bool native_tiling, Common::ThreadWorker* workers) {
    constexpr size_t MinBytes = 512 * 1024;
    constexpr u32 MaxBands = 4;
    const size_t pitch = size_t{width} * bytes_per_pixel;
    const size_t host_bytes = pitch * height;
    if (depth != 1 || block_depth != 0 || host_bytes < MinBytes) {
        return false;
    }
    const u32 band_rows = GOB_SIZE_Y << block_height;
    const u32 block_rows = Common::DivCeil(height, band_rows);
    const u32 bands = (std::min)({MaxBands, block_rows,
                                  static_cast<u32>(host_bytes / (MinBytes / 2))});
    if (bands < 2) {
        return false;
    }
    const size_t stride = size_t{Common::AlignUpLog2(width, stride_alignment)} * bytes_per_pixel;
    const size_t band_bytes =
        Common::DivCeil(stride, size_t{GOB_SIZE_X}) * (size_t{GOB_SIZE} << block_height);
    const size_t guest_bytes = size_t{block_rows} * band_bytes;
    if (input.size() < guest_bytes || output.size() < host_bytes) {
        return false;
    }
    const auto run_band = [=](u32 band) {
        const u32 first = block_rows * band / bands;
        const u32 last = block_rows * (band + 1) / bands;
        const u32 first_row = first * band_rows;
        const u32 rows = (std::min)(height, last * band_rows) - first_row;
        UnswizzleTexture(output.subspan(size_t{first_row} * pitch, size_t{rows} * pitch),
                         input.subspan(size_t{first} * band_bytes), bytes_per_pixel, width, rows,
                         1, block_height, 0, stride_alignment, native_tiling);
    };
    auto& pool = workers ? *workers : GetThreadWorkers();
    std::latch completed{static_cast<std::ptrdiff_t>(bands - 1)};
    for (u32 band = 0; band + 1 < bands; ++band) {
        if (!pool.TryQueueWork([&run_band, &completed, band] {
                SCOPE_EXIT {
                    completed.count_down();
                };
                run_band(band);
            })) {
            run_band(band);
            completed.count_down();
        }
    }
    run_band(bands - 1);
    completed.wait();
    return true;
}
} // Anonymous namespace

bool UnswizzleWorker::TryUnswizzle(std::span<u8> output, std::span<const u8> input,
                                   u32 bytes_per_pixel, u32 width, u32 height, u32 depth,
                                   u32 block_height, u32 block_depth, u32 stride_alignment,
                                   bool native_tiling, bool verify, Common::ThreadWorker* workers) {
    const size_t pitch = size_t{width} * bytes_per_pixel;
    const size_t host_bytes = pitch * height;
    static const bool fast = [] { const char* v = std::getenv("EDEN_NX2_FAST_UNSWIZZLE"); return v == nullptr || v[0] != '0'; }();
    if (fast) {
        return TryUnswizzleBands(output, input, bytes_per_pixel, width, height, depth,
                                 block_height, block_depth, stride_alignment, native_tiling,
                                 workers);
    }
    if (!native_tiling || depth != 1 || block_depth != 0 ||
        host_bytes < ParallelUnswizzleMinBytes || (pitch & 3) == 0) {
        return false;
    }
    const u32 band_rows = GOB_SIZE_Y << block_height;
    const u32 first_rows = height / 2 / band_rows * band_rows;
    if (first_rows == 0) {
        return false;
    }
    const size_t stride = size_t{Common::AlignUpLog2(width, stride_alignment)} * bytes_per_pixel;
    const size_t band_bytes =
        Common::DivCeil(stride, size_t{GOB_SIZE_X}) * (size_t{GOB_SIZE} << block_height);
    const size_t guest_bytes = Common::DivCeil(height, band_rows) * band_bytes;
    if (guest_bytes > 32 * 1024 * 1024 || guest_bytes > 2 * host_bytes ||
        input.size() < guest_bytes || output.size() < host_bytes) {
        return false;
    }
    owned.resize_destructive(guest_bytes);
    std::memcpy(owned.data(), input.data(), guest_bytes);
    const std::span<const u8> snapshot{owned};
    const size_t guest_split = first_rows / band_rows * band_bytes;
    const size_t host_split = first_rows * pitch;
    std::latch completed{1};
    auto& pool = workers ? *workers : GetThreadWorkers();
    if (!pool.TryQueueWork([=, &completed] {
            SCOPE_EXIT {
                completed.count_down();
            };
            UnswizzleTexture(output.first(host_split), snapshot.first(guest_split), bytes_per_pixel,
                             width, first_rows, 1, block_height, 0, stride_alignment,
                             native_tiling);
        })) {
        return false;
    }
    {
        SCOPE_EXIT {
            completed.wait();
        };
        UnswizzleTexture(output.subspan(host_split, host_bytes - host_split),
                         snapshot.subspan(guest_split), bytes_per_pixel, width, height - first_rows,
                         1, block_height, 0, stride_alignment, native_tiling);
    }
    if (verify) {
        std::vector<u8> reference(host_bytes);
        UnswizzleTexture(reference, snapshot, bytes_per_pixel, width, height, depth, block_height,
                         block_depth, stride_alignment, native_tiling);
        if (std::memcmp(reference.data(), output.data(), host_bytes) != 0) {
            LOG_ERROR(HW_GPU,
                      "Nx2Perf unswizzle_workers: output differs for {}x{} bpp={} native={}", width,
                      height, bytes_per_pixel, native_tiling);
        }
        std::memcpy(output.data(), reference.data(), host_bytes);
    }
    return true;
}

} // namespace Tegra::Texture
