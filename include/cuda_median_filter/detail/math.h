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

#include <type_traits>

namespace quxflux
{
  template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr T int_div_ceil(const T x, const T y)
  {
    // https://stackoverflow.com/a/2745086/1130270

    return x == 0 ? 0 : 1 + ((x - 1) / y);
  }

  template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr T pad(const T x, const T padding)
  {
    return int_div_ceil(x, padding) * padding;
  }
}  // namespace quxflux
