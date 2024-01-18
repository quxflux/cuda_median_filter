#pragma once

#include <cuda_median_filter/detail/image_source_target.h>
#include <cuda_median_filter/detail/math.h>
#include <cuda_median_filter/detail/pitched_array_accessor.h>
#include <cuda_median_filter/detail/static_for.h>

#include <cstdint>

namespace quxflux::detail
{
  template<typename T, std::int32_t BlockSize, std::int32_t ApronWidth, std::int32_t ApronHeight, typename ImageSource>
  constexpr void load_apron(const pitched_array_accessor<T, detail::mutable_access> dst, const ImageSource img_source,
                            const point<std::int32_t>& apron_origin, const point<std::int32_t>& thread_idx)
  {
    constexpr std::int32_t num_load_ops_y = int_div_ceil(ApronHeight, BlockSize);
    constexpr std::int32_t num_load_ops_x = int_div_ceil(ApronWidth, BlockSize);

    static_for_2d<num_load_ops_y, num_load_ops_x>([&](const auto mult) {
      const std::int32_t apron_y = thread_idx.y + mult.y * BlockSize;
      const std::int32_t apron_x = thread_idx.x + mult.x * BlockSize;

      const std::int32_t pixel_y = apron_origin.y + apron_y;
      const std::int32_t pixel_x = apron_origin.x + apron_x;

      T pixel_value{};

      if (inside_bounds<std::int32_t>({pixel_x, pixel_y}, img_source.bounds()))
        pixel_value = img_source.get({pixel_x, pixel_y});

      if (inside_bounds<std::int32_t>({apron_x, apron_y}, {ApronWidth, ApronHeight}))
        dst.set(pixel_value, {apron_x, apron_y});
    });
  }
}  // namespace quxflux::detail
