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