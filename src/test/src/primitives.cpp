#include <cuda_median_filter/detail/primitives.h>

#include <gtest/gtest.h>

namespace quxflux
{
  TEST(inside_bounds, returns_false_for_empty_bounds)
  {
    const bounds<int> b{0, 0};
    EXPECT_FALSE(inside_bounds({0, 0}, b));
    EXPECT_FALSE(inside_bounds({1, 1}, b));
  }

  TEST(inside_bounds, with_signed_type)
  {
    const bounds<int> b{10, 20};
    EXPECT_TRUE(inside_bounds({0, 0}, b));
    EXPECT_TRUE(inside_bounds({9, 19}, b));
    EXPECT_TRUE(inside_bounds({5, 5}, b));

    EXPECT_FALSE(inside_bounds({-1, 0}, b));
    EXPECT_FALSE(inside_bounds({0, -1}, b));
    EXPECT_FALSE(inside_bounds({0, 20}, b));
    EXPECT_FALSE(inside_bounds({10, 0}, b));
  }

  TEST(inside_bounds, with_unsigned_type)
  {
    const bounds<unsigned> b{10, 20};
    EXPECT_TRUE(inside_bounds({0, 0}, b));
    EXPECT_TRUE(inside_bounds({9, 19}, b));
    EXPECT_TRUE(inside_bounds({5, 5}, b));

    EXPECT_FALSE(inside_bounds({10, 0}, b));
    EXPECT_FALSE(inside_bounds({0, 20}, b));
  }
}