#pragma once

#include <cuda_median_filter/detail/math.h>
#include <cuda_median_filter/detail/primitives.h>

#include <cstdint>

namespace quxflux::detail::image_filter_config
{
  constexpr bounds<std::int32_t> calculate_block_bounds(const std::int32_t block_size, const std::int32_t simd_width)
  {
    return {simd_width * block_size, block_size};
  }

  constexpr bounds<std::int32_t> calculate_block_bounds_with_apron(const std::int32_t block_size,
                                                                   const std::int32_t filter_size,
                                                                   const std::int32_t simd_width)
  {
    const std::int32_t filter_radius = filter_size / 2;
    const auto bounds_without_apron = calculate_block_bounds(block_size, simd_width);
    return {bounds_without_apron.width + 2 * filter_radius, bounds_without_apron.height + 2 * filter_radius};
  }

  template<typename PixelType>
  constexpr std::int32_t calculate_shared_buf_row_pitch(const std::int32_t block_size, const std::int32_t filter_size,
                                                        const std::int32_t simd_width)
  {
    const auto block_bounds = calculate_block_bounds_with_apron(block_size, filter_size, simd_width);
    return pad(block_bounds.width * std::int32_t{sizeof(PixelType)}, simd_width);
  }

  template<typename PixelType>
  constexpr std::int32_t calculate_required_shared_buf_size(const std::int32_t block_size,
                                                            const std::int32_t filter_size,
                                                            const std::int32_t simd_width)
  {
    return calculate_shared_buf_row_pitch<PixelType>(block_size, filter_size, simd_width) *
           calculate_block_bounds_with_apron(block_size, filter_size, simd_width).height;
  }
}  // namespace quxflux::detail::image_filter_config
