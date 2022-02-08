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

#include <shared/image.h>

#include <impls/opencv/opencv_wrap.h>

namespace quxflux
{
  template<typename T>
  cv::Mat create_mat_view_for_image(const image<T>& img)
  {
    const auto bounds = img.bounds();

    return cv::Mat(bounds.height, bounds.width, cv::DataType<T>::type, const_cast<byte*>(img.data_ptr()),
                   static_cast<std::size_t>(img.row_pitch_in_bytes()));
  }
}  // namespace quxflux
