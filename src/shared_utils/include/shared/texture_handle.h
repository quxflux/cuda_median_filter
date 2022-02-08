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

namespace quxflux
{
  template<typename T>
  class texture_handle
  {
  public:
    texture_handle() = default;

    texture_handle(const byte* device_ptr, const bounds<std::int32_t>& bounds, std::int32_t row_pitch_in_bytes)
    {
      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = cast_to_cuda_ptr(device_ptr);
      resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
      resDesc.res.pitch2D.width = bounds.width;
      resDesc.res.pitch2D.height = bounds.height;
      resDesc.res.pitch2D.pitchInBytes = row_pitch_in_bytes;

      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      cuda_call(&cudaCreateTextureObject, &tex_, &resDesc, &texDesc, nullptr);
    }

    texture_handle(texture_handle&&) = default;
    texture_handle& operator=(texture_handle&&) = default;

    texture_handle(const texture_handle&) = delete;
    texture_handle& operator=(const texture_handle&) = delete;

    explicit operator bool() const noexcept { return tex_ != 0; }

    operator cudaTextureObject_t() const noexcept { return tex_; }

    ~texture_handle()
    {
      [[maybe_unused]] const cudaError_t r = cudaDestroyTextureObject(tex_);

      assert(r == cudaSuccess && "cudaDestroyTextureObject failed!");
    }

  private:
    cudaTextureObject_t tex_{};
  };

}  // namespace quxflux
