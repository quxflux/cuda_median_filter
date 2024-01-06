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
#include <shared/image.h>

#include <cuda_median_filter/detail/cuda/wrap_cuda.h>
#include <cuda_median_filter/detail/primitives.h>

#include <cstdint>

namespace quxflux
{
  template<typename T>
  using gpu_image = image<T, struct CudaImage>;

  template<typename T>
  gpu_image<T> make_gpu_image(const bounds<int32_t>& bounds)
  {
    auto [ptr, pitch] = make_unique_device_pitched(sizeof(T) * static_cast<size_t>(bounds.width),
                                                   static_cast<size_t>(bounds.height));
    return {std::shared_ptr{std::move(ptr)}, bounds, pitch};
  }
}  // namespace quxflux
