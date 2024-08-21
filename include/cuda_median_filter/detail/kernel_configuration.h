#pragma once

#include <cuda_median_filter/detail/math.h>

#include <cstdint>

namespace quxflux::detail
{
  template<typename>
  struct no_vectorization
  {
    static inline constexpr std::int32_t items_per_thread = 1;
  };

  template<typename T, std::int32_t FilterSize, std::int32_t BlockSize,
           template<typename> typename VectorizationTraits = no_vectorization>
  struct median_2d_configuration
  {
    using value_type = T;

    static inline constexpr std::int32_t items_per_thread = VectorizationTraits<T>::items_per_thread;

    static inline constexpr bool vectorize = items_per_thread > 1;

    static inline constexpr std::int32_t block_size = BlockSize;

    static inline constexpr std::int32_t filter_size = FilterSize;
    static inline constexpr std::int32_t filter_radius = FilterSize / 2;

    static inline constexpr std::int32_t num_pixels_x = items_per_thread * BlockSize;
    static inline constexpr std::int32_t num_pixels_y = BlockSize;

    static inline constexpr std::int32_t shared_buf_row_pitch = pad(
      (num_pixels_x + 2 * filter_radius) * std::int32_t{sizeof(T)}, items_per_thread);
    static inline constexpr std::int32_t shared_buf_size = shared_buf_row_pitch * (num_pixels_y + 2 * filter_radius);
  };
}  // namespace quxflux::detail