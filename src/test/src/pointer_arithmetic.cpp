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

#include <cuda_median_filter/detail/pointer_arithmetic.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>

namespace quxflux
{
  TEST(calculate_pitched_address, returns_expected_result_for_byte_sized_pointer)
  {
    const std::byte* const base_address = reinterpret_cast<const std::byte*>(0xBADF00D);

    EXPECT_EQ(calculate_pitched_address(base_address, 16, 0, 0), base_address);
    EXPECT_EQ(calculate_pitched_address(base_address, 16, 3, 0), std::next(base_address, 3));
    EXPECT_EQ(calculate_pitched_address(base_address, 16, 16, 0), std::next(base_address, 16));
    EXPECT_EQ(calculate_pitched_address(base_address, 16, 0, 1), std::next(base_address, 16));
    EXPECT_EQ(calculate_pitched_address(base_address, 16, 1, 1), std::next(base_address, 17));
    EXPECT_EQ(calculate_pitched_address(base_address, 16, 5, 1), std::next(base_address, 21));
  }

  TEST(calculate_pitched_address, returns_expected_result_for_multi_byte_type_pointer)
  {
    const std::byte* const base_address = reinterpret_cast<const std::byte*>(0xBADF00D);

    EXPECT_EQ(calculate_pitched_address<uint32_t>(base_address, 16, 0, 0), base_address);
    EXPECT_EQ(calculate_pitched_address<uint32_t>(base_address, 16, 3, 0), std::next(base_address, 3 * 4));
    EXPECT_EQ(calculate_pitched_address<uint32_t>(base_address, 16, 16, 0), std::next(base_address, 16 * 4));
    EXPECT_EQ(calculate_pitched_address<uint32_t>(base_address, 16, 0, 1), std::next(base_address, 16));
    EXPECT_EQ(calculate_pitched_address<uint32_t>(base_address, 16, 1, 1), std::next(base_address, 16 + 1 * 4));
    EXPECT_EQ(calculate_pitched_address<uint32_t>(base_address, 16, 5, 1), std::next(base_address, 16 + 5 * 4));
  }
}  // namespace quxflux