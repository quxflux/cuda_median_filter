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

#include <gmock/gmock-matchers.h>

#include <cuda_median_filter/detail/image_source_target.h>

#include <shared/image.h>
#include <shared/index2d_range.h>

#include <algorithm>
#include <iomanip>

namespace quxflux
{
  template<typename T>
  std::ostream& operator<<(std::ostream& os, const bounds<T>& b)
  {
    return os << "[width = " << b.width << ", height = " << b.height << "]";
  }

  template<typename T>
  class images_are_equal_matcher
  {
  public:
    images_are_equal_matcher() = default;
    images_are_equal_matcher(const image<T>& image) : expected_(image) {}

    using is_gtest_matcher = void;

    bool MatchAndExplain(const image<T>& img, ::testing::MatchResultListener* os) const
    {
      if (img.bounds() != expected_.bounds())
      {
        *os << "bounds are mismatched. Expected " << expected_.bounds() << ", got " << img.bounds();

        return false;
      }

      const auto bounds = img.bounds();

      const pitched_array_image_source<T> expected_img(expected_.data_ptr(), bounds, expected_.row_pitch_in_bytes());
      const pitched_array_image_source<T> actual_img(img.data_ptr(), bounds, img.row_pitch_in_bytes());

      const auto index_range = index2d_range(std::array{bounds.width, bounds.height});

      const auto num_mismatched = std::count_if(index_range.begin(), index_range.end(), [&](const auto idx) {
        const auto [x, y] = idx;
        return expected_img.get({x, y}) != actual_img.get({x, y});
      });

      if (num_mismatched)
      {
        *os << num_mismatched << " pixels are deviating from expected image (" << std::setprecision(2)
            << static_cast<float>(num_mismatched) / static_cast<float>(bounds.width * bounds.height) * 100.f << " %)";

        return false;
      }

      return true;
    }

    void DescribeTo(std::ostream* os) const { *os << "is equal to expected image"; }

    void DescribeNegationTo(std::ostream* os) const { *os << "is not equal to expected image"; }

  private:
    image<T> expected_;
  };

  template<typename T>
  auto is_equal_to_image(const image<T>& img)
  {
    return images_are_equal_matcher<T>(img);
  }
}  // namespace quxflux
