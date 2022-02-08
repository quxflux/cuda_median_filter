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

#include <cuda_median_filter/detail/primitives.h>

namespace quxflux
{
  template<typename T = byte, typename ByteT>
  constexpr ByteT* calculate_pitched_address(ByteT* const base_address, const std::int32_t row_pitch, std::int32_t x,
                                             std::int32_t y)
  {
    return base_address + y * row_pitch + x * std::int32_t{sizeof(T)};
  }
}  // namespace quxflux
