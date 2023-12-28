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

#include <cuda_median_filter/detail/cuda_wrap.h>
#include <cuda_median_filter/detail/pointer_arithmetic.h>
#include <cuda_median_filter/detail/primitives.h>

#include <cstdint>
#include <cstring>

namespace quxflux
{
  namespace detail
  {
    struct const_access
    {};

    struct mutable_access
    {};
  }  // namespace detail

  template<typename T, typename Access = detail::mutable_access>
  class pitched_array_accessor
  {
    static inline bool constexpr is_mutable_access = std::is_same_v<Access, detail::mutable_access>;

  public:
    using ptr_t = std::conditional_t<is_mutable_access, byte, const byte>*;

    constexpr pitched_array_accessor(ptr_t const data_ptr, const std::int32_t row_pitch)
      : data_ptr_(data_ptr), row_pitch_(row_pitch)
    {}

    [[nodiscard]] constexpr T get(const point<std::int32_t>& coord) const
    {
      T r;
      memcpy(&r, calculate_pitched_address<T>(data_ptr_, row_pitch_, coord.x, coord.y), sizeof(T));
      return r;
    }

    template<typename Q = Access, typename = std::enable_if_t<std::is_same_v<Q, detail::mutable_access>>>
    constexpr void set(const T& value, const point<std::int32_t>& coord) const
    {
      memcpy(calculate_pitched_address<T>(data_ptr_, row_pitch_, coord.x, coord.y), &value, sizeof(T));
    }

    [[nodiscard]] constexpr ptr_t data_ptr() const { return data_ptr_; }
    [[nodiscard]] constexpr std::int32_t row_pitch() const { return row_pitch_; }

  private:
    ptr_t data_ptr_;
    std::int32_t row_pitch_;
  };
}  // namespace quxflux
