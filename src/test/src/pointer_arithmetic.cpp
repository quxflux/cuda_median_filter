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