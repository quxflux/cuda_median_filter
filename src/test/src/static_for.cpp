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

#include <cuda_median_filter/detail/static_for.h>

#include <gmock/gmock.h>

#include <array>
#include <cstdint>
#include <vector>

namespace quxflux
{
  namespace
  {
    template<std::int32_t N>
    std::vector<std::int32_t> materialize()
    {
      std::vector<std::int32_t> result;
      static_for<N>([&](const std::int32_t i) { result.push_back(i); });
      return result;
    }

    template<std::int32_t X, std::int32_t Y>
    std::vector<std::tuple<std::int32_t, std::int32_t>> materialize_2d()
    {
      std::vector<std::tuple<std::int32_t, std::int32_t>> result;
      static_for_2d<X, Y>([&](const auto i) { result.emplace_back(i.x, i.y); });
      return result;
    }

  }  // namespace

  TEST(static_for, does_not_call_f_when_n_is_zero)
  {
    EXPECT_THAT(materialize<0>(), testing::IsEmpty());
  }

  TEST(static_for, calls_f_with_expected_values)
  {
    EXPECT_THAT(materialize<7>(), testing::ElementsAre(0, 1, 2, 3, 4, 5, 6));
  }

  TEST(static_for_2d, does_not_call_f_when_x_or_y_is_zero)
  {
    EXPECT_THAT((materialize_2d<0, 0>()), testing::IsEmpty());
    EXPECT_THAT((materialize_2d<1, 0>()), testing::IsEmpty());
    EXPECT_THAT((materialize_2d<0, 1>()), testing::IsEmpty());
  }

  TEST(static_for_2d, calls_f_with_expected_values)
  {
    EXPECT_THAT((materialize_2d<2, 3>()),
                testing::ElementsAreArray(std::to_array<std::tuple<std::int32_t, std::int32_t>>(
                  {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {2, 1}})));
  }
}  // namespace quxflux
