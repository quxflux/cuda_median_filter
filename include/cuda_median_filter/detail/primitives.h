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

namespace quxflux
{
  template<typename... Ops>
  constexpr bool eager_logical_and(Ops... ops)
  {
    return (ops && ...);
  }

  template<typename T>
  struct bounds
  {
    T width{};
    T height{};

    constexpr bool operator==(const bounds&) const = default;
  };

  template<typename T>
  struct point
  {
    T x{};
    T y{};

    constexpr bool operator==(const point&) const = default;
  };

  template<typename T>
  constexpr bool inside_bounds(const point<T>& p, const bounds<T>& bounds)
  {
    return eager_logical_and(p.x >= T{0}, p.y >= T{0}, p.x < bounds.width, p.y < bounds.height);
  }
}  // namespace quxflux
