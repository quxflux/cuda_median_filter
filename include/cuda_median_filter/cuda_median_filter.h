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

#include <cuda_median_filter/detail/cuda/median_filter_impl.h>
#include <cuda_median_filter/detail/cuda/texture_image_source.h>
#include <cuda_median_filter/detail/cuda/wrap_cuda.h>
#include <cuda_median_filter/detail/image_source_target.h>
#include <cuda_median_filter/detail/primitives.h>

namespace quxflux
{
  struct median_2d_expert_settings;

  template<typename T, std::int32_t FilterSize, typename ExpertSettings = median_2d_expert_settings>
  void median_2d_async(const cudaTextureObject_t src_texture, void* const dst, const std::int32_t dst_pitch_in_bytes,
                       const std::int32_t width, const std::int32_t height, const cudaStream_t stream = 0)
  {
    const bounds<std::int32_t> img_bounds{width, height};

    texture_image_source<T> texture_image_source(src_texture, img_bounds);
    pitched_array_image_target<T> array2d_image_target(static_cast<byte*>(dst), img_bounds, dst_pitch_in_bytes);

    detail::median_2d_async<FilterSize, ExpertSettings>(texture_image_source, array2d_image_target, stream);
  }

  template<typename T, std::int32_t FilterSize, typename ExpertSettings = median_2d_expert_settings>
  void median_2d_async(const void* const src, const std::int32_t src_pitch_in_bytes, void* const dst,
                       const std::int32_t dst_pitch_in_bytes, const std::int32_t width, const std::int32_t height,
                       const cudaStream_t stream = 0)
  {
    const bounds<std::int32_t> img_bounds{width, height};

    pitched_array_image_source<T> array2d_image_source(static_cast<const byte*>(src), img_bounds, src_pitch_in_bytes);
    pitched_array_image_target<T> array2d_image_target(static_cast<byte*>(dst), img_bounds, dst_pitch_in_bytes);

    detail::median_2d_async<FilterSize, ExpertSettings>(array2d_image_source, array2d_image_target, stream);
  }

  struct median_2d_expert_settings
  {
    static inline constexpr std::int32_t block_size = 16;
    static inline constexpr std::int32_t max_filter_size_allowed_for_vectorization = 7;
  };
}  // namespace quxflux
