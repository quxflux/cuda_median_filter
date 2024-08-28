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

#include <cuda_median_filter/detail/math.h>

#include <gtest/gtest.h>

#include <cstdint>

namespace quxflux
{
  TEST(int_div_ceil, returns_expected_result)
  {
    EXPECT_EQ(int_div_ceil(0, 1), 0);
    EXPECT_EQ(int_div_ceil(1, 1), 1);
    EXPECT_EQ(int_div_ceil(2, 1), 2);
    EXPECT_EQ(int_div_ceil(3, 2), 2);
    EXPECT_EQ(int_div_ceil(4, 2), 2);
    EXPECT_EQ(int_div_ceil(5, 2), 3);
    EXPECT_EQ(int_div_ceil(127, 16), 8);
  }

  TEST(pad, returns_expected_result)
  {
    EXPECT_EQ(pad(0, 1), 0);
    EXPECT_EQ(pad(1, 1), 1);
    EXPECT_EQ(pad(2, 1), 2);
    EXPECT_EQ(pad(3, 2), 4);
    EXPECT_EQ(pad(4, 2), 4);
    EXPECT_EQ(pad(5, 2), 6);
    EXPECT_EQ(pad(127, 16), 128);
  }
}  // namespace quxflux