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

#include <shared/cuda/gpu_image.h>
#include <shared/image.h>

namespace quxflux
{
  template<typename PixelT>
  void transfer(const gpu_image<PixelT>& img_src, image<PixelT>& img_dst, cudaStream_t stream = 0)
  {
    cuda_call(&cudaMemcpy2DAsync, img_dst.data_ptr(), static_cast<std::size_t>(img_dst.row_pitch_in_bytes()),
              img_src.data_ptr(), static_cast<std::size_t>(img_src.row_pitch_in_bytes()),
              static_cast<std::size_t>(img_src.bounds().width) * sizeof(PixelT),
              static_cast<std::size_t>(img_src.bounds().height), cudaMemcpyDefault, stream);
  }

  template<typename PixelT>
  void transfer(const image<PixelT>& img_src, gpu_image<PixelT>& img_dst, cudaStream_t stream = 0)
  {
    cuda_call(&cudaMemcpy2DAsync, img_dst.data_ptr(), static_cast<std::size_t>(img_dst.row_pitch_in_bytes()),
              cast_to_cuda_ptr(img_src.data_ptr()), static_cast<std::size_t>(img_src.row_pitch_in_bytes()),
              static_cast<std::size_t>(img_src.bounds().width) * sizeof(PixelT),
              static_cast<std::size_t>(img_src.bounds().height), cudaMemcpyDefault, stream);
  }
}  // namespace quxflux
