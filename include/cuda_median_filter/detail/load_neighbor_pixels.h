#pragma once

#include <cuda_median_filter/detail/math.h>
#include <cuda_median_filter/detail/primitives.h>
#include <cuda_median_filter/detail/static_for.h>

#include <cstdint>

namespace quxflux::detail
{
  template<typename T, typename KernelConfig, typename ImageDestination, typename ImageSource>
  constexpr void load_neighbor_pixels(ImageDestination& dst, const ImageSource img_source,
                                      const point<std::int32_t>& group_idx, const point<std::int32_t>& local_idx)
  {
    constexpr std::int32_t neighborhood_height = KernelConfig::num_pixels_y + 2 * KernelConfig::filter_radius;
    constexpr std::int32_t neighborhood_width = KernelConfig::num_pixels_x + 2 * KernelConfig::filter_radius;

    constexpr std::int32_t num_load_ops_y = int_div_ceil(neighborhood_height, KernelConfig::block_size);
    constexpr std::int32_t num_load_ops_x = int_div_ceil(neighborhood_width, KernelConfig::block_size);

    using idx_2d = point<std::int32_t>;

    const idx_2d apron_origin = {KernelConfig::num_pixels_x * group_idx.x - KernelConfig::filter_radius,
                                 KernelConfig::num_pixels_y * group_idx.y - KernelConfig::filter_radius};

    static_for_2d<num_load_ops_y, num_load_ops_x>([&](const auto mult) {
      const idx_2d apron_idx = {local_idx.x + mult.x * KernelConfig::block_size,
                                local_idx.y + mult.y * KernelConfig::block_size};

      const idx_2d pixel_idx = {apron_origin.x + apron_idx.x, apron_origin.y + apron_idx.y};

      T pixel_value{};

      if (inside_bounds(pixel_idx, img_source.bounds()))
        pixel_value = img_source.get(pixel_idx);

      if (inside_bounds(apron_idx, {neighborhood_width, neighborhood_height}))
        dst.set(pixel_value, apron_idx);
    });
  }
}  // namespace quxflux::detail
