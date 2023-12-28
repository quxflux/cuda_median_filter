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

#include <shared/cuda/util.h>

#include <cuda_median_filter/detail/cuda/wrap_cuda.h>
#include <cuda_median_filter/detail/primitives.h>

#include <cstdint>

namespace quxflux
{
  template<typename T>
  class gpu_image : public detail::bounded_image<T>
  {
    using base = detail::bounded_image<T>;

  public:
    constexpr gpu_image() = default;

    explicit gpu_image(const bounds<std::int32_t>& bounds) : base(bounds) { resize(bounds); }

    void resize(const bounds<std::int32_t>& bounds)
    {
      if (bounds.width > base::bounds_.width || bounds.height > base::bounds_.height)
      {
        std::tie(data_, row_pitch_bytes_) = make_unique_device_pitched(
          static_cast<std::size_t>(bounds.width * std::int32_t{sizeof(T)}), static_cast<std::size_t>(bounds.height));
      }

      base::bounds_ = bounds;
    }

    constexpr std::int32_t row_pitch_in_bytes() const { return row_pitch_bytes_; }

    [[nodiscard]] constexpr byte* data() { return data_.get(); }
    [[nodiscard]] constexpr const byte* data() const { return data_.get(); }

    [[nodiscard]] constexpr explicit operator bool() const
    {
      return data_ && base::bounds_.width > 0 && base::ounds_.height > 0 && row_pitch_bytes_ >= base::bounds_.width;
    }

  private:
    unique_pitched_device_ptr data_;

    std::int32_t row_pitch_bytes_ = 0;
  };
}  // namespace quxflux
