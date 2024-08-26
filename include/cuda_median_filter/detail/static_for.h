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

#include <cstdint>
#include <type_traits>
#include <utility>

namespace quxflux
{
  namespace detail
  {
    template<typename T>
    struct advance;

    template<std::int32_t I>
    struct advance<std::integral_constant<std::int32_t, I>>
    {
      using type = std::integral_constant<std::int32_t, I + 1>;
    };

    template<std::int32_t X, std::int32_t Y, std::int32_t NumCols>
    struct constant_index2d_iterator
    {
      static constexpr std::int32_t x = X;
      static constexpr std::int32_t y = Y;
    };

    template<std::int32_t X, std::int32_t Y, std::int32_t NumCols>
    struct advance<constant_index2d_iterator<X, Y, NumCols>>
    {
      static constexpr bool next_y = X + 1 >= NumCols;

      using type = constant_index2d_iterator<next_y ? 0 : X + 1, next_y ? Y + 1 : Y, NumCols>;
    };

    template<typename Current, typename End, typename F>
    constexpr void static_for_impl(F&& f)
    {
      if constexpr (std::is_same_v<Current, End>)
        return;
      else
      {
        f(Current{});
        static_for_impl<typename advance<Current>::type, End, F>(std::forward<F>(f));
      }
    }
  }  // namespace detail

  template<std::int32_t ExtentY, std::int32_t ExtentX, typename F>
  constexpr void static_for_2d(F&& f)
  {
    static_assert(ExtentX >= 0 && ExtentY >= 0, "ExtentX and ExtentY must be greater than 0");

    using start = detail::constant_index2d_iterator<0, 0, ExtentX>;
    using end = detail::constant_index2d_iterator<0, ExtentY, ExtentX>;

    if constexpr (ExtentY * ExtentX > 0)
      detail::static_for_impl<start, end>(std::forward<F>(f));
  }

  template<std::int32_t N, typename F>
  constexpr void static_for(F&& f)
  {
    static_assert(N >= 0, "N must be greater or equal to 0");

    using start = std::integral_constant<std::int32_t, 0>;
    using end = std::integral_constant<std::int32_t, N>;

    detail::static_for_impl<start, end>(std::forward<F>(f));
  }
}  // namespace quxflux
