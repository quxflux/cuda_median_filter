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

#include <cuda_median_filter/detail/cuda/wrap_cuda.h>
#include <cuda_median_filter/detail/kernel_configuration.h>
#include <cuda_median_filter/detail/load_apron.h>
#include <cuda_median_filter/detail/math.h>
#include <cuda_median_filter/detail/pitched_array_accessor.h>
#include <cuda_median_filter/detail/pointer_arithmetic.h>
#include <cuda_median_filter/detail/primitives.h>
#include <cuda_median_filter/detail/static_for.h>

#include <sorting_network_cpp/sorting_network.h>

#include <array>
#include <cstdint>

namespace quxflux::detail
{
  template<typename T>
  struct vectorization_for_byte_sized_values
  {
    static inline constexpr std::int32_t items_per_thread = (sizeof(T) == 1 ? 4 : 1);
  };

  namespace kernels
  {
    template<typename KernelConfig, typename ImageSource, typename ImageTarget>
    __global__ void median_2d(const ImageSource img_source, const ImageTarget dst)
    {
      using T = typename ImageSource::value_type;
      using config = KernelConfig;

      const std::int32_t tidx_y = static_cast<std::int32_t>(threadIdx.y);
      const std::int32_t tidx_x = static_cast<std::int32_t>(threadIdx.x);

      extern __shared__ byte shared_buf_data[];

      const pitched_array_accessor<T> shared_buf(shared_buf_data, config::shared_buf_row_pitch);

      {
        const std::int32_t apron_origin_y = config::num_pixels_y * blockIdx.y - config::filter_radius;
        const std::int32_t apron_origin_x = config::num_pixels_x * blockIdx.x - config::filter_radius;

        load_apron<T, config::block_size, config::apron_width, config::apron_height>(
          shared_buf, img_source, point<std::int32_t>{apron_origin_x, apron_origin_y},
          point<std::uint32_t>{threadIdx.x, threadIdx.y});
      }
      __syncthreads();

      using sorting_t = std::conditional_t<config::vectorize, unsigned int, T>;

      sorting_t filtered_value;

      // gather all neighbor pixels and calculate the median value
      {
        std::array<sorting_t, config::filter_size * config::filter_size> local_neighborhood_pixels;
        auto it = local_neighborhood_pixels.begin();

        static_for_2d<config::filter_size, config::filter_size>([&](const auto idx) {
          const std::int32_t dy = idx.y - config::filter_radius;
          const std::int32_t dx = idx.x - config::filter_radius;
          const std::int32_t apron_y = tidx_y + config::filter_radius + dy;

          if constexpr (config::vectorize)
          {
            const std::int32_t apron_x = tidx_x * config::items_per_thread + config::filter_radius + dx;

            unsigned int r;

            const bool is_aligned_access = apron_x % 4 == 0;

            if (is_aligned_access)
            {
              r = reinterpret_cast<const unsigned int&>(
                *calculate_pitched_address<>(shared_buf_data, config::shared_buf_row_pitch, apron_x, apron_y));
            } else
            {
              auto& v = reinterpret_cast<uchar4&>(r);
              v.x = shared_buf.get({apron_x + 0, apron_y});
              v.y = shared_buf.get({apron_x + 1, apron_y});
              v.z = shared_buf.get({apron_x + 2, apron_y});
              v.w = shared_buf.get({apron_x + 3, apron_y});
            }

            *(it++) = r;
          } else
          {
            const std::int32_t apron_x = tidx_x + config::filter_radius + dx;
            *(it++) = shared_buf.get({apron_x, apron_y});
          }
        });

        constexpr sorting_net::sorting_network<config::filter_size * config::filter_size> sorting_net;

        if constexpr (config::vectorize)
        {
          sorting_net(local_neighborhood_pixels.begin(), [](auto& a, auto& b) {
            const auto a_cpy = a;

            a = __vminu4(a, b);
            b = __vmaxu4(a_cpy, b);
          });
        } else
        {
          sorting_net(local_neighborhood_pixels.begin(), [](auto& a, auto& b) {
            const auto a_cpy = a;

            a = std::min(a, b);
            b = std::max(a_cpy, b);
          });
        }

        filtered_value = *(local_neighborhood_pixels.begin() + (config::filter_size * config::filter_size) / 2);
      }

      const std::int32_t global_thread_idx_y = tidx_y + config::num_pixels_y * blockIdx.y;
      const std::int32_t global_thread_idx_x = tidx_x * config::items_per_thread +
                                               config::num_pixels_x * (std::int32_t)blockIdx.x;

      if (inside_bounds<std::int32_t>({global_thread_idx_x, global_thread_idx_y}, img_source.bounds()))
      {
        if constexpr (config::vectorize)
        {
          reinterpret_cast<unsigned int&>(*calculate_pitched_address<>(
            dst.data_ptr(), dst.row_pitch(), global_thread_idx_x, global_thread_idx_y)) = filtered_value;
        } else
        {
          dst.set(filtered_value, {global_thread_idx_x, global_thread_idx_y});
        }
      }
    }
  }  // namespace kernels

  template<std::int32_t FilterSize, typename ExpertSettings, typename ImageSource, typename ImageTarget>
  void median_2d_async(const ImageSource img_src, const ImageTarget img_dst, const cudaStream_t stream)
  {
    static_assert(FilterSize % 2 == 1, "Filter size must be odd");

    using T = typename ImageSource::value_type;

    constexpr std::int32_t N = ExpertSettings::block_size;

    static constexpr auto vectorization_allowed = FilterSize <= ExpertSettings::max_filter_size_allowed_for_vectorization;

    using filter_config =
      std::conditional_t<vectorization_allowed,
                         median_2d_configuration<T, FilterSize, N, vectorization_for_byte_sized_values>,
                         median_2d_configuration<T, FilterSize, N, no_vectorization>>;

    const dim3 block_size(N, N);
    const dim3 grid_size(int_div_ceil(img_dst.bounds().width, filter_config::num_pixels_x),
                         int_div_ceil(img_dst.bounds().height, filter_config::num_pixels_y));

    kernels::median_2d<filter_config>
      <<<grid_size, block_size, filter_config::shared_buf_size, stream>>>(img_src, img_dst);
  }
}  // namespace quxflux::detail
