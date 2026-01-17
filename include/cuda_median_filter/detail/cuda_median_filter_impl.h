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
    __device__ inline auto filter_serial(const pitched_array_accessor<T> data)
    {
      constexpr auto n_filter_elements = FilterSize * FilterSize;
      constexpr auto filter_radius = FilterSize / 2;

      const auto [local_idx, block_idx] = get_thread_coordinates();

      std::array<T, n_filter_elements> local_neighborhood_pixels;
      auto it = local_neighborhood_pixels.begin();

      static_for_2d<FilterSize, FilterSize>([&](const auto idx) {
        const std::int32_t dy = idx.y - filter_radius;
        const std::int32_t dx = idx.x - filter_radius;

        const idx_2d apron_idx = {local_idx.x * SimdWidth + filter_radius + dx, local_idx.y + filter_radius + dy};
        *(it++) = data.get(apron_idx);
      });

      constexpr sorting_net::sorting_network<n_filter_elements> sorting_net;

      sorting_net(local_neighborhood_pixels.begin(), [](auto& a, auto& b) {
        const auto a_cpy = a;

        a = std::min(a, b);
        b = std::max(a_cpy, b);
      });

      return *(local_neighborhood_pixels.begin() + n_filter_elements / 2);
    }

    template<std::int32_t FilterSize, std::int32_t SimdWidth, typename T>
    __device__ inline auto filter_simd(const pitched_array_accessor<T> data)
    {
      constexpr auto n_filter_elements = FilterSize * FilterSize;
      constexpr auto filter_radius = FilterSize / 2;

      const auto [local_idx, block_idx] = get_thread_coordinates();

      std::array<unsigned int, n_filter_elements> local_neighborhood_pixels;
      auto it = local_neighborhood_pixels.begin();

      static_for_2d<FilterSize, FilterSize>([&](const auto idx) {
        const std::int32_t dy = idx.y - filter_radius;
        const std::int32_t dx = idx.x - filter_radius;

        const idx_2d apron_idx = {local_idx.x * SimdWidth + filter_radius + dx, local_idx.y + filter_radius + dy};

        static_assert(SimdWidth == 4);
        unsigned int r;

        const bool is_aligned_access = apron_idx.x % 4 == 0;

        if (is_aligned_access)
        {
          r = reinterpret_cast<const unsigned int&>(
            *calculate_pitched_address<>(data.data_ptr(), data.row_pitch(), apron_idx.x, apron_idx.y));
        } else
        {
          auto& [x, y, z, w] = reinterpret_cast<uchar4&>(r);
          x = data.get({apron_idx.x + 0, apron_idx.y});
          y = data.get({apron_idx.x + 1, apron_idx.y});
          z = data.get({apron_idx.x + 2, apron_idx.y});
          w = data.get({apron_idx.x + 3, apron_idx.y});
        }

        *(it++) = r;
      });

      constexpr sorting_net::sorting_network<n_filter_elements> sorting_net;

      sorting_net(local_neighborhood_pixels.begin(), [](auto& a, auto& b) {
        const auto a_cpy = a;

        a = __vminu4(a, b);
        b = __vmaxu4(a_cpy, b);
      });

      return *(local_neighborhood_pixels.begin() + n_filter_elements / 2);
    }


    template<std::int32_t FilterSize, std::int32_t BlockSize, std::int32_t SimdWidth, typename ImageSource,
             typename ImageTarget>
    __global__ void median_2d(const ImageSource img_source, const ImageTarget dst)
    {
      namespace config = image_filter_config;

      using T = typename ImageSource::value_type;


      const auto [local_idx, block_idx] = get_thread_coordinates();

      extern __shared__ std::byte shared_buf_data[];
      const pitched_array_accessor<T> shared_buf(
        shared_buf_data, config::calculate_shared_buf_row_pitch<T>(BlockSize, FilterSize, SimdWidth));

      load_neighbor_pixels<T, load_neighbor_params{.block_size = BlockSize,
                                                   .filter_size = FilterSize,
                                                   .local_bounds = config::calculate_block_bounds(BlockSize,
                                                                                                  SimdWidth)}>  //
        (shared_buf, img_source, block_idx, local_idx);
      __syncthreads();

      constexpr bool vectorize = SimdWidth > 1;
      using sorting_t = std::conditional_t<vectorize, unsigned int, T>;

      sorting_t filtered_value;
      if constexpr (vectorize)
      {
        filtered_value = filter_simd<FilterSize, SimdWidth>(shared_buf);
      } else
      {
        filtered_value = filter_serial<FilterSize, SimdWidth>(shared_buf);
      }

      constexpr auto local_bounds = config::calculate_block_bounds(BlockSize, SimdWidth);

      const idx_2d global_idx = {local_idx.x * SimdWidth + local_bounds.width * block_idx.x,
                                 local_idx.y + local_bounds.height * block_idx.y};

      if (inside_bounds(global_idx, img_source.bounds()))
      {
        if constexpr (vectorize)
        {
          reinterpret_cast<unsigned int&>(
            *calculate_pitched_address<>(dst.data_ptr(), dst.row_pitch(), global_idx.x, global_idx.y)) = filtered_value;
        } else
        {
          dst.set(filtered_value, global_idx);
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
