// This file is part of the cuda_median_filter (https://github.com/quxflux/cuda_median_filter).
// Copyright (c) 2022 Lukas Riebel.
//
// cuda_median_filter is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// cuda_median_filter is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <cuda_median_filter/detail/cuda_wrap.h>
#include <cuda_median_filter/detail/image_filter_config.h>
#include <cuda_median_filter/detail/load_neighbor_pixels.h>
#include <cuda_median_filter/detail/math.h>
#include <cuda_median_filter/detail/pitched_array_accessor.h>
#include <cuda_median_filter/detail/pointer_arithmetic.h>
#include <cuda_median_filter/detail/primitives.h>
#include <cuda_median_filter/detail/static_for.h>

#include <sorting_network_cpp/sorting_network.h>

#include <array>
#include <cstring>
#include <cstdint>
#include <type_traits>

namespace quxflux::detail
{
  namespace kernels
  {
    using idx_2d = point<int32_t>;

    __device__ inline std::tuple<point<int32_t>, point<int32_t>> get_thread_coordinates()
    {
      return {{static_cast<int32_t>(threadIdx.x), static_cast<int32_t>(threadIdx.y)},
              {static_cast<int32_t>(blockIdx.x), static_cast<int32_t>(blockIdx.y)}};
    }

    template<std::int32_t FilterSize, std::int32_t SimdWidth, typename T>
    __device__ std::array<T, SimdWidth> filter(const pitched_array_accessor<T> data)
    {
      static_assert(SimdWidth == 1 || SimdWidth == 4);
      using element_type = std::array<T, SimdWidth>;

      constexpr auto n_filter_elements = FilterSize * FilterSize;
      constexpr auto filter_radius = FilterSize / 2;

      const auto [local_idx, block_idx] = get_thread_coordinates();

      std::array<element_type, n_filter_elements> local_neighborhood_pixels;
      auto it = local_neighborhood_pixels.begin();

      static_for_2d<FilterSize, FilterSize>([&](const auto idx) {
        const std::int32_t dy = idx.y - filter_radius;
        const std::int32_t dx = idx.x - filter_radius;

        const idx_2d apron_idx = {local_idx.x * SimdWidth + filter_radius + dx, local_idx.y + filter_radius + dy};

        element_type r;

        const bool is_aligned_access = apron_idx.x % SimdWidth == 0 &&
                                       (SimdWidth * sizeof(T) == 32 || SimdWidth * sizeof(T) == 64);

        if (is_aligned_access)
        {
          std::memcpy(r.data(),
                      calculate_pitched_address<>(data.data_ptr(), data.row_pitch(), apron_idx.x, apron_idx.y),
                      sizeof(element_type));
        } else
        {
          for (int i = 0; i < SimdWidth; ++i)
            r[i] = data.get({apron_idx.x + i, apron_idx.y});
        }

        *(it++) = r;
      });

      constexpr sorting_net::sorting_network<n_filter_elements> sorting_net;

      sorting_net(local_neighborhood_pixels.begin(), [](element_type& a, element_type& b) {
        const element_type a_cpy = a;

        for (int i = 0; i < SimdWidth; ++i)
        {
          a[i] = std::min(a_cpy[i], b[i]);
          b[i] = std::max(a_cpy[i], b[i]);
        }
      });

      return *(local_neighborhood_pixels.begin() + n_filter_elements / 2);
    }

    template<std::int32_t FilterSize, std::int32_t BlockSize, std::int32_t SimdWidth, typename ImageSource,
             typename ImageTarget>
    __global__ void median_2d(const ImageSource img_source, const ImageTarget dst)
    {
      namespace config = image_filter_config;

      using T = typename ImageSource::value_type;
      static constexpr auto local_bounds = config::calculate_block_bounds(BlockSize, SimdWidth);

      const auto [local_idx, block_idx] = get_thread_coordinates();

      extern __shared__ std::byte shared_buf_data[];
      const pitched_array_accessor<T> shared_buf(
        shared_buf_data, config::calculate_shared_buf_row_pitch<T>(BlockSize, FilterSize, SimdWidth));

      load_neighbor_pixels<
        T, load_neighbor_params{.block_size = BlockSize, .filter_size = FilterSize, .local_bounds = local_bounds}>  //
        (shared_buf, img_source, block_idx, local_idx);
      __syncthreads();

      const auto filtered_value = filter<FilterSize, SimdWidth>(shared_buf);

      const idx_2d global_idx = {local_idx.x * SimdWidth + local_bounds.width * block_idx.x,
                                 local_idx.y + local_bounds.height * block_idx.y};

      if (inside_bounds(global_idx, img_source.bounds()))
      {
        if constexpr (sizeof(T) * SimdWidth == 32 || sizeof(T) * SimdWidth == 64)
        {
          std::memcpy(calculate_pitched_address<>(dst.data_ptr(), dst.row_pitch(), global_idx.x, global_idx.y),
                      filtered_value.data(), SimdWidth * sizeof(T));
        } else
        {
          for (int i = 0; i < SimdWidth; ++i)
            dst.set(filtered_value[i], {global_idx.x + i, global_idx.y + 0});
        }
      }
    }
  }  // namespace kernels

  template<std::int32_t FilterSize, typename ExpertSettings, typename ImageSource, typename ImageTarget>
  void median_2d_async(const ImageSource img_src, const ImageTarget img_dst, const cudaStream_t stream)
  {
    static_assert(FilterSize % 2 == 1, "Filter size must be odd");

    using T = typename ImageSource::value_type;

    // if one pixel is one byte large, we can use SIMD Video Instructions to process 4 pixels at once, see
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simd-video-instructions
    constexpr std::int32_t simd_width = sizeof(T) == 1 ? 4 : 1;

    constexpr std::int32_t N = ExpertSettings::block_size;
    constexpr std::int32_t items_per_thread = FilterSize <= ExpertSettings::max_filter_size_allowed_for_vectorization
                                                ? simd_width
                                                : 1;
    const auto block_bounds = image_filter_config::calculate_block_bounds(N, items_per_thread);

    const dim3 block_size(N, N);
    const dim3 grid_size(int_div_ceil(img_dst.bounds().width, block_bounds.width),
                         int_div_ceil(img_dst.bounds().height, block_bounds.height));

    constexpr auto required_shared_buf_size = image_filter_config::calculate_required_shared_buf_size<T>(
      N, FilterSize, items_per_thread);
    kernels::median_2d<FilterSize, N, items_per_thread>
      <<<grid_size, block_size, required_shared_buf_size, stream>>>(img_src, img_dst);
  }
}  // namespace quxflux::detail
