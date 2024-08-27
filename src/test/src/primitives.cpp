#include <cuda_median_filter/detail/primitives.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>

namespace quxflux
{
  namespace
  {
    class implicitly_bool_convertible
    {
    public:
      constexpr explicit implicitly_bool_convertible(bool value) : m_value(value) {}

      constexpr operator bool()
      {
        m_got_converted = true;
        return m_value;
      }

      constexpr bool got_converted() const { return m_got_converted; }

    private:
      bool m_got_converted = false;
      bool m_value = false;
    };
  }  // namespace

  TEST(eager_logical_and, returns_expected_result)
  {
    EXPECT_TRUE(eager_logical_and(true, true));
    EXPECT_FALSE(eager_logical_and(true, false));
    EXPECT_FALSE(eager_logical_and(false, true));
    EXPECT_FALSE(eager_logical_and(false, false));
  }

  TEST(eager_logical_and, does_evaluate_all_operands_even_if_one_is_false)
  {
    implicitly_bool_convertible a{true}, b{false}, c{true};
    EXPECT_FALSE(eager_logical_and(a, b, c));
    EXPECT_TRUE(std::ranges::all_of(std::array{a, b, c}, &implicitly_bool_convertible::got_converted));
  }

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
}  // namespace quxflux