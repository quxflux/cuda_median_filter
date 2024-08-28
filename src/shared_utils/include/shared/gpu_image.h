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

#include <shared/cuda_util.h>

#include <cuda_median_filter/detail/cuda_wrap.h>
#include <cuda_median_filter/detail/primitives.h>

#include <cstdint>

namespace quxflux
{
  template<typename T>
  class gpu_image
  {
  public:
    gpu_image() = default;

    gpu_image(const bounds<std::int32_t>& bounds) { resize(bounds); }

    void resize(const bounds<std::int32_t>& bounds)
    {
      if (bounds.width > bounds_.width || bounds.height > bounds_.height)
      {
        std::tie(data_, row_pitch_bytes_) = make_unique_device_pitched(
          static_cast<std::size_t>(bounds.width * std::int32_t{sizeof(T)}), static_cast<std::size_t>(bounds.height));
      }

      bounds_ = bounds;
    }

    auto bounds() const noexcept { return bounds_; }
    std::int32_t row_pitch_in_bytes() const { return row_pitch_bytes_; }

    std::byte* data() { return data_.get(); }
    const std::byte* data() const { return data_.get(); }

    explicit operator bool() const
    {
      return data_ && bounds_.width > 0 && bounds_.height > 0 && row_pitch_bytes_ >= bounds_.width;
    }

  private:
    unique_pitched_device_ptr data_;

    quxflux::bounds<std::int32_t> bounds_;
    std::int32_t row_pitch_bytes_ = 0;
  };
}  // namespace quxflux
