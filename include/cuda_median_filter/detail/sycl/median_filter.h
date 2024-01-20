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

  namespace detail
  {
    constexpr point<std::int32_t> to_idx2d(const auto& sycl_id)
    {
      return point<std::int32_t>::from_any(sycl_id[1], sycl_id[0]);
    }

    constexpr auto* get_ptr(auto& accessor_like)
    {
      auto* t = accessor_like.template get_multi_ptr<sycl::access::decorated::no>().get();

      if constexpr (std::is_const_v<std::remove_pointer_t<decltype(t)>>)
        return reinterpret_cast<const byte*>(t);
      else
        return reinterpret_cast<byte*>(t);
    }
  }  // namespace detail

  template<std::int32_t FilterSize, typename ExpertSettings = sycl_median_2d_expert_settings,
           typename T = std::void_t<>, typename SourceAllocator = std::void_t<>,
           typename TargetAllocator = std::void_t<>>
  void median_2d_async(sycl::buffer<T, 2, SourceAllocator>& src, sycl::buffer<T, 2, TargetAllocator>& dst,
                       sycl::queue& queue, const std::optional<std::int32_t> width_if_src_is_pitched = std::nullopt)
  {
    static_assert(FilterSize % 2 == 1, "Filter size must be odd");

    using namespace detail;

    using config = median_2d_configuration<T, FilterSize, ExpertSettings::block_size, no_vectorization>;

    static constexpr auto n_filter_elements = config::filter_size * config::filter_size;

    const auto src_pitch_bytes = static_cast<int32_t>(src.get_range()[1] * sizeof(T));
    const auto actual_width = width_if_src_is_pitched.value_or(static_cast<int32_t>(src.get_range()[1]));

    const bounds<std::int32_t> img_bounds{actual_width, static_cast<int32_t>(src.get_range()[0])};

    if (img_bounds.width * img_bounds.height == 0)
      return;

    queue.submit([&](sycl::handler& handler) {
      sycl::accessor input{src, sycl::read_only};
      sycl::accessor output{dst, sycl::write_only};

      handler.require(input);
      handler.require(output);

      auto local_mem = sycl::local_accessor<byte, 1>(config::shared_buf_size, handler);

      const auto nd_range = sycl::nd_range<2>{
        {static_cast<size_t>(int_div_ceil(img_bounds.height, config::block_size) * config::block_size),
         static_cast<size_t>(int_div_ceil(img_bounds.width, config::block_size) * config::block_size)},
        {config::block_size, config::block_size},
      };

      handler.parallel_for(nd_range, [=](const sycl::nd_item<2> idx) {
        T filtered_value = {};

        const auto local_idx = to_idx2d(idx.get_local_id());

        const auto shared_accessor = pitched_array_accessor<T>{get_ptr(local_mem), config::shared_buf_row_pitch};
        load_apron<T, config>(shared_accessor,
                              pitched_array_image_source<T>{get_ptr(input), img_bounds, src_pitch_bytes},
                              to_idx2d(idx.get_group().get_group_id()), local_idx);

        group_barrier(idx.get_group());

        std::array<T, n_filter_elements> local_neighborhood_pixels;
        auto it = local_neighborhood_pixels.begin();

        static_for_2d<config::filter_size, config::filter_size>([&](const auto offset) {
          *(it++) = shared_accessor.get({local_idx.x + offset.x, local_idx.y + offset.y});
        });

        group_barrier(idx.get_group());

        constexpr sorting_net::sorting_network<n_filter_elements> sorting_net;

        sorting_net(local_neighborhood_pixels.begin(), [](auto& a, auto& b) {
          const auto a_cpy = a;

          a = std::min(a, b);
          b = std::max(a_cpy, b);
        });

        filtered_value = *std::midpoint(local_neighborhood_pixels.begin(),
                                        local_neighborhood_pixels.begin() + n_filter_elements);

        output[idx.get_global_id()] = filtered_value;
      });
    });
  }
}  // namespace quxflux
