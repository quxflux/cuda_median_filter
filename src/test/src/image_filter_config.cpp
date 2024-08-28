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

#include <cuda_median_filter/detail/image_filter_config.h>

#include <gtest/gtest.h>

#include <cstdint>

namespace quxflux::detail
{
  TEST(kernel_configuration, get_block_bounds)
  {
    EXPECT_EQ(image_filter_config::calculate_block_bounds(16, 1), (bounds{16 * 1, 16}));
    EXPECT_EQ(image_filter_config::calculate_block_bounds(16, 4), (bounds{16 * 4, 16}));
  }

  TEST(kernel_configuration, get_block_bounds_with_apron)
  {
    EXPECT_EQ(image_filter_config::calculate_block_bounds_with_apron(16, 5, 1), (bounds{16 * 1 + 4, 16 + 4}));
    EXPECT_EQ(image_filter_config::calculate_block_bounds_with_apron(16, 5, 4), (bounds{16 * 4 + 4, 16 + 4}));
  }

  TEST(kernel_configuration, calculate_shared_buf_row_pitch)
  {
    EXPECT_EQ(image_filter_config::calculate_shared_buf_row_pitch<uint8_t>(16, 5, 1), 16 + 4);
    EXPECT_EQ(image_filter_config::calculate_shared_buf_row_pitch<uint8_t>(16, 5, 4), (16 * 4) + 4);
    EXPECT_EQ(image_filter_config::calculate_shared_buf_row_pitch<uint8_t>(16, 7, 4), 72);
    EXPECT_EQ(image_filter_config::calculate_shared_buf_row_pitch<uint32_t>(16, 7, 1), (16 + 6) * 4);
  }

  TEST(kernel_configuration, calculate_required_shared_buf_size)
  {
    EXPECT_EQ(image_filter_config::calculate_required_shared_buf_size<uint8_t>(16, 7, 1), 484);
    EXPECT_EQ(image_filter_config::calculate_required_shared_buf_size<uint8_t>(16, 7, 4), 1584);
    EXPECT_EQ(image_filter_config::calculate_required_shared_buf_size<uint32_t>(16, 7, 1), 1936);
  }
}  // namespace quxflux::detail
