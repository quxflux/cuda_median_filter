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

#include <cuda_median_filter/detail/image_source_target.h>

#include <shared/image.h>
#include <shared/index2d_range.h>

#include <algorithm>
#include <array>

namespace quxflux
{
  template<std::int32_t FilterSize, typename T>
  void filter_image(const image<T>& src, image<T>& dst)
  {
    assert(src.bounds() == dst.bounds());

    pitched_array_image_source<T> img_src(src.data_ptr(), src.bounds(), src.row_pitch_in_bytes());
    pitched_array_image_target<T> img_dst(dst.data_ptr(), dst.bounds(), dst.row_pitch_in_bytes());

    for (const auto [x, y] : index2d_range(std::array{src.bounds().width, src.bounds().height}))
    {
      std::array<T, FilterSize * FilterSize> buf{};
      auto it = buf.begin();

      for (const auto [filter_x, filter_y] : index2d_range(std::array{FilterSize, FilterSize}))
      {
        const auto dx = x + filter_x - FilterSize / 2;
        const auto dy = y + filter_y - FilterSize / 2;

        if (inside_bounds({dx, dy}, src.bounds()))
        {
          *(it++) = img_src.get({dx, dy});
        }
      }

      const auto median = buf.begin() + buf.size() / 2;
      std::nth_element(buf.begin(), median, buf.end());

      img_dst.set(*median, {x, y});
    }
  }
}  // namespace quxflux
