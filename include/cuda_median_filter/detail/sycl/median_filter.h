// This file is part of the cuda_median_filter (https://github.com/quxflux/cuda_median_filter).
// Copyright (c) 2024 Lukas Riebel.
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

#include <cuda_median_filter/detail/image_source_target.h>
#include <cuda_median_filter/detail/kernel_configuration.h>
#include <cuda_median_filter/detail/load_apron.h>
#include <cuda_median_filter/detail/pitched_array_accessor.h>

#include <sorting_network_cpp/sorting_network.h>

#include <sycl/sycl.hpp>

#include <cstdint>
#include <optional>

namespace quxflux
{
  struct sycl_median_2d_expert_settings
  {
    static inline constexpr std::int32_t block_size = 16;
  };

  template<std::int32_t FilterSize, typename ExpertSettings = sycl_median_2d_expert_settings,
           typename T = std::void_t<>, typename SourceAllocator = std::void_t<>,
           typename TargetAllocator = std::void_t<>>
  void median_2d_async(sycl::buffer<T, 2, SourceAllocator>& src, sycl::buffer<T, 2, TargetAllocator>& dst,
                       sycl::queue& queue, const std::optional<std::int32_t> width_if_src_is_pitched = std::nullopt)
  {
    static_assert(FilterSize % 2 == 1, "Filter size must be odd");

    using config = detail::median_2d_configuration<T, FilterSize, ExpertSettings::block_size, detail::no_vectorization>;


    const auto src_pitch_bytes = static_cast<int32_t>(src.get_range()[1] * sizeof(T));
    const auto actual_width = width_if_src_is_pitched.value_or(static_cast<int32_t>(src.get_range()[1]));

    const bounds<std::int32_t> img_bounds{actual_width, static_cast<int32_t>(src.get_range()[0])};

    queue.submit([&](sycl::handler& handler) {
      sycl::accessor input{src, sycl::read_only};
      sycl::accessor output{dst, sycl::write_only};

      handler.require(input);
      handler.require(output);

      auto local_mem = sycl::local_accessor<byte, 1>(config::shared_buf_size, handler);

      const auto nd_range = sycl::nd_range<2>{
        {static_cast<size_t>(img_bounds.height), static_cast<size_t>(img_bounds.width)},
        {config::block_size, config::block_size},
      };

      handler.parallel_for(nd_range, [=](const sycl::nd_item<2> idx) {
        T filtered_value = {};

        const auto local_index_y = static_cast<int32_t>(idx.get_local_id(0));
        const auto local_index_x = static_cast<int32_t>(idx.get_local_id(1));

        const auto shared_accessor = pitched_array_accessor<T>{
          local_mem.get_multi_ptr<sycl::access::decorated::no>().get(),
          config::shared_buf_row_pitch,
        };

        {
          const auto block_idx_y = idx.get_group().get_group_id()[0];
          const auto block_idx_x = idx.get_group().get_group_id()[1];
          const std::int32_t apron_origin_y = config::num_pixels_y * block_idx_y - config::filter_radius;
          const std::int32_t apron_origin_x = config::num_pixels_x * block_idx_x - config::filter_radius;

          detail::load_apron<T, config::block_size, config::apron_width, config::apron_height>(
            shared_accessor,
            pitched_array_image_source<T>{input.template get_multi_ptr<sycl::access::decorated::no>().get(), img_bounds,
                                          src_pitch_bytes},
            point<std::int32_t>{apron_origin_x, apron_origin_y}, point<std::int32_t>{local_index_x, local_index_y});
        }

        group_barrier(idx.get_group());

        std::array<T, config::filter_size * config::filter_size> local_neighborhood_pixels;
        auto it = local_neighborhood_pixels.begin();

        static_for_2d<config::filter_size, config::filter_size>([&](const auto idx_local) {
          *(it++) = shared_accessor.get({local_index_x + idx_local.x, local_index_y + idx_local.y});
        });

        group_barrier(idx.get_group());

        constexpr sorting_net::sorting_network<FilterSize * FilterSize> sorting_net;

        sorting_net(local_neighborhood_pixels.begin(), [](auto& a, auto& b) {
          const auto a_cpy = a;

          a = std::min(a, b);
          b = std::max(a_cpy, b);
        });

        filtered_value = *(local_neighborhood_pixels.begin() + (FilterSize * FilterSize) / 2);

        output[idx.get_global_id()] = filtered_value;
      });
    });
  }
}  // namespace quxflux
