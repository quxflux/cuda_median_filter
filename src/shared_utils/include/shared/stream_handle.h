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

namespace quxflux
{
  class stream_handle
  {
  public:
    stream_handle(const unsigned int flags = cudaStreamNonBlocking)
    {
      cuda_call(&cudaStreamCreateWithFlags, &stream_, flags);
    }

    stream_handle(stream_handle&&) = default;
    stream_handle& operator=(stream_handle&&) = default;

    stream_handle(const stream_handle&) = delete;
    stream_handle& operator=(const stream_handle&) = delete;

    void synchronize() const { cuda_call(&cudaStreamSynchronize, stream_); }

    explicit operator bool() const noexcept { return stream_ != 0; }

    operator cudaStream_t() const noexcept { return stream_; }

    auto get_flags() const
    {
      unsigned int flags = 0;
      cuda_call(&cudaStreamGetFlags, stream_, &flags);
      return flags;
    }

    ~stream_handle()
    {
      if (!!*this)
      {
        [[maybe_unused]] const cudaError_t r = cudaStreamDestroy(stream_);
        assert(r == cudaSuccess && "CUDA free failed!");

        stream_ = 0;
      }
    }

  private:
    cudaStream_t stream_ = 0;
  };
}  // namespace quxflux
