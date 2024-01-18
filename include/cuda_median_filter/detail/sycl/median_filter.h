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
#include <cuda_median_filter/detail/load_apron.h>

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
                       sycl::queue& queue, const std::optional<std::int32_t> width = std::nullopt)
  {
    const bounds<std::int32_t> img_bounds{
      width.value_or(static_cast<int32_t>(src.get_range()[0])),
      static_cast<int32_t>(src.get_range()[1]),
    };

    queue.submit([&](sycl::handler& handler) {
      sycl::accessor input{src, sycl::read_only};
      sycl::accessor output{dst, sycl::write_only};

      handler.require(input);
      handler.require(output);

      handler.parallel_for(
        sycl::range<2>{src.get_range()[0], static_cast<size_t>(width.value_or(src.get_range()[1]))},
        [=](const sycl::id<2> idx) {
          T filtered_value = {};

          std::array<T, FilterSize * FilterSize> local_neighborhood_pixels;
          auto it = local_neighborhood_pixels.begin();

          static_for_2d<FilterSize, FilterSize>([&](const auto idx_local) {
            const std::int32_t dy = idx_local.y - FilterSize / 2;
            const std::int32_t dx = idx_local.x - FilterSize / 2;

            const auto global_y = static_cast<int32_t>(idx.get(0)) + dy;
            const auto global_x = static_cast<int32_t>(idx.get(1)) + dx;

            if (global_x >= 0 && global_x < img_bounds.width && global_y >= 0 && global_y < img_bounds.height)
            {
              *(it++) = input[sycl::id<2>{static_cast<size_t>(global_y), static_cast<size_t>(global_x)}];
            } else
            {
              *(it++) = T{};
            }
          });

          constexpr sorting_net::sorting_network<FilterSize * FilterSize> sorting_net;

          sorting_net(local_neighborhood_pixels.begin(), [](auto& a, auto& b) {
            const auto a_cpy = a;

            a = std::min(a, b);
            b = std::max(a_cpy, b);
          });

          filtered_value = *(local_neighborhood_pixels.begin() + (FilterSize * FilterSize) / 2);

          output[idx] = filtered_value;
        });
    });
  }
}  // namespace quxflux
