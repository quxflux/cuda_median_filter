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

#include <cuda_median_filter/detail/cuda/wrap_cuda.h>
#include <cuda_median_filter/detail/image_source_target.h>
#include <cuda_median_filter/detail/math.h>
#include <cuda_median_filter/detail/primitives.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace quxflux
{
  template<typename T>
  class image : public detail::bounded_image<T>
  {
    using base = detail::bounded_image<T>;
    static_assert(std::is_arithmetic_v<T>, "Only arithmetic types are supported");

    static inline constexpr std::int32_t DefaultAlignment = 1024;

  public:
    constexpr image() = default;
    image(const bounds<std::int32_t>& bounds, const std::int32_t row_pitch_in_bytes = 0)
    {
      resize(bounds, row_pitch_in_bytes);
    }
    image(const std::shared_ptr<byte> pre_allocated_data, const bounds<std::int32_t>& bounds,
          const std::int32_t row_pitch_in_bytes)
      : detail::bounded_image<T>(bounds), row_pitch_in_bytes_(row_pitch_in_bytes), data_(pre_allocated_data)
    {}

    constexpr image(const image&) = default;
    constexpr image(image&&) = default;
    constexpr image& operator=(const image&) = default;
    constexpr image& operator=(image&&) = default;

    void resize(const bounds<std::int32_t>& bounds, const std::int32_t row_pitch_in_bytes = 0)
    {
      base::bounds_ = bounds;
      row_pitch_in_bytes_ = row_pitch_in_bytes >= base::bounds_.width * std::int32_t{sizeof(T)}
                              ? row_pitch_in_bytes
                              : int_div_ceil(std::int32_t{sizeof(T)} * base::bounds_.width, DefaultAlignment) *
                                  DefaultAlignment;
      data_ = std::shared_ptr<byte>(
        static_cast<byte*>(::operator new(static_cast<std::size_t>(base::bounds_.height * row_pitch_in_bytes_))));
    }

    explicit constexpr operator bool() const
    {
      return base::bounds_.width > 0 && base::bounds_.height > 0 &&
             row_pitch_in_bytes_ >= sizeof(T) * base::bounds_.width && data_;
    }

    [[nodiscard]] constexpr std::int32_t row_pitch_in_bytes() const { return row_pitch_in_bytes_; }

    [[nodiscard]] constexpr byte* data_ptr() { return data_.get(); }
    [[nodiscard]] constexpr const byte* data_ptr() const { return data_.get(); }

    [[nodiscard]] constexpr byte* row_data_ptr(const std::int32_t row)
    {
      return data_.get() + row_pitch_in_bytes_ * row;
    }
    [[nodiscard]] constexpr const byte* row_data_ptr(const std::int32_t row) const
    {
      return data_.get() + row_pitch_in_bytes_ * row;
    }

  private:
    std::int32_t row_pitch_in_bytes_ = 0;

    std::shared_ptr<byte> data_;
  };
}  // namespace quxflux
