#pragma once

#include <cuda_median_filter/detail/image_source_target.h>
#include <cuda_median_filter/detail/math.h>
#include <cuda_median_filter/detail/pitched_array_accessor.h>
#include <cuda_median_filter/detail/static_for.h>

#include <cstdint>

namespace quxflux::detail
{
  template<typename T, typename KernelConfig, typename ImageSource>
  constexpr void load_apron(const pitched_array_accessor<T, detail::mutable_access> dst, const ImageSource img_source,
                            const point<std::int32_t>& group_idx, const point<std::int32_t>& local_idx)
  {
    constexpr std::int32_t num_load_ops_y = int_div_ceil(KernelConfig::apron_height, KernelConfig::block_size);
    constexpr std::int32_t num_load_ops_x = int_div_ceil(KernelConfig::apron_width, KernelConfig::block_size);

    const point<std::int32_t> apron_origin = {KernelConfig::num_pixels_x * group_idx.x - KernelConfig::filter_radius,
                                              KernelConfig::num_pixels_y * group_idx.y - KernelConfig::filter_radius};

    static_for_2d<num_load_ops_y, num_load_ops_x>([&](const auto mult) {
      const std::int32_t apron_y = local_idx.y + mult.y * KernelConfig::block_size;
      const std::int32_t apron_x = local_idx.x + mult.x * KernelConfig::block_size;

      const std::int32_t pixel_y = apron_origin.y + apron_y;
      const std::int32_t pixel_x = apron_origin.x + apron_x;

      T pixel_value{};

      if (inside_bounds<std::int32_t>({pixel_x, pixel_y}, img_source.bounds()))
        pixel_value = img_source.get({pixel_x, pixel_y});

      if (inside_bounds<std::int32_t>({apron_x, apron_y}, {KernelConfig::apron_width, KernelConfig::apron_height}))
        dst.set(pixel_value, {apron_x, apron_y});
    });
  }
}  // namespace quxflux::detail
