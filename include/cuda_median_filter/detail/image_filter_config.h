#pragma once

#include <cuda_median_filter/detail/math.h>
#include <cuda_median_filter/detail/primitives.h>

#include <cstdint>

namespace quxflux::detail::image_filter_config
{
  constexpr bounds<std::int32_t> get_block_bounds(const std::int32_t block_size, const std::int32_t items_per_thread)
  {
    return {items_per_thread * block_size, block_size};
  }

  constexpr bounds<std::int32_t> get_block_bounds_with_apron(const std::int32_t block_size,
                                                             const std::int32_t filter_size,
                                                             const std::int32_t items_per_thread)
  {
    const std::int32_t filter_radius = filter_size / 2;
    const auto bounds_without_apron = get_block_bounds(block_size, items_per_thread);
    return {bounds_without_apron.width + 2 * filter_radius, bounds_without_apron.height + 2 * filter_radius};
  }

  template<typename PixelType>
  constexpr std::int32_t get_shared_buf_row_pitch(const std::int32_t block_size, const std::int32_t filter_size,
                                                  const std::int32_t items_per_thread)
  {
    const auto block_bounds = get_block_bounds_with_apron(block_size, filter_size, items_per_thread);
    return pad(block_bounds.width * std::int32_t{sizeof(PixelType)}, items_per_thread);
  }

  template<typename PixelType>
  constexpr std::int32_t get_shared_buf_size(const std::int32_t block_size, const std::int32_t filter_size,
                                             const std::int32_t items_per_thread)
  {
    return get_shared_buf_row_pitch<PixelType>(block_size, filter_size, items_per_thread) *
           get_block_bounds_with_apron(block_size, filter_size, items_per_thread).height;
  }
}  // namespace quxflux::detail::image_filter_config
