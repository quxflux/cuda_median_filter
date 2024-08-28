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

#include <cuda_median_filter/detail/math.h>
#include <cuda_median_filter/detail/primitives.h>
#include <cuda_median_filter/detail/static_for.h>

#include <cstdint>

namespace quxflux::detail
{
  struct load_neighbor_params
  {
    std::int32_t block_size{};
    std::int32_t filter_size{};
    bounds<std::int32_t> local_bounds{};
  };

  template<typename T, load_neighbor_params StaticParams, typename ImageDestination, typename ImageSource>
  constexpr void load_neighbor_pixels(ImageDestination& dst, const ImageSource img_source,
                                      const point<std::int32_t>& group_idx, const point<std::int32_t>& local_idx)
  {
    constexpr auto block_size = StaticParams.block_size;
    constexpr auto filter_radius = StaticParams.filter_size / 2;

    constexpr bounds<std::int32_t> local_bounds = StaticParams.local_bounds;
    constexpr bounds<std::int32_t> local_bounds_with_apron = {local_bounds.width + 2 * filter_radius,
                                                              local_bounds.height + 2 * filter_radius};

    constexpr auto num_load_ops_y = int_div_ceil(local_bounds_with_apron.height, block_size);
    constexpr auto num_load_ops_x = int_div_ceil(local_bounds_with_apron.width, block_size);

    using idx_2d = point<std::int32_t>;

    const idx_2d apron_origin = {local_bounds.width * group_idx.x - filter_radius,
                                 local_bounds.height * group_idx.y - filter_radius};

    static_for_2d<num_load_ops_y, num_load_ops_x>([&](const auto mult) {
      const idx_2d apron_idx = {local_idx.x + mult.x * block_size, local_idx.y + mult.y * block_size};
      const idx_2d pixel_idx = {apron_origin.x + apron_idx.x, apron_origin.y + apron_idx.y};

      T pixel_value{};

      if (inside_bounds(pixel_idx, img_source.bounds()))
        pixel_value = img_source.get(pixel_idx);

      if (inside_bounds(apron_idx, local_bounds_with_apron))
        dst.set(pixel_value, apron_idx);
    });
  }
}  // namespace quxflux::detail
