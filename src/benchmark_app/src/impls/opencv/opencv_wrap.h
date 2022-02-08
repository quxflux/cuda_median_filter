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

#include <type_traits>
#include <utility>

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#endif

#include <opencv2/imgproc.hpp>

#if __has_include(<opencv2/cudafilters.hpp>)
#include <opencv2/cudafilters.hpp>
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace quxflux
{
  template<typename T, typename = void>
  struct cv_mat_supports_type : std::false_type
  {};

  template<typename T>
  struct cv_mat_supports_type<T, std::void_t<decltype(cv::DataType<T>::type)>> : std::true_type
  {};
}  // namespace quxflux
