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

#include <array>
#include <random>

namespace quxflux
{
  template<typename T, typename RandomEngine = std::default_random_engine>
  void fill_image_random(image<T>& img, RandomEngine rd = RandomEngine{42})
  {
    const pitched_array_image_target<T> img_dst(img.data_ptr(), img.bounds(), img.row_pitch_in_bytes());

    for (const auto [x, y] : index2d_range(std::array{img.bounds().width, img.bounds().height}))
    {
      T random_value{};

      if constexpr (std::is_floating_point_v<T>)
        random_value = std::uniform_real_distribution<T>(T{0}, T{1})(rd);
      else
        random_value = static_cast<T>(
          std::uniform_int_distribution<std::uint64_t>(T{0}, std::numeric_limits<T>::max())(rd));

      img_dst.set(random_value, {x, y});
    }
  }
}  // namespace quxflux
