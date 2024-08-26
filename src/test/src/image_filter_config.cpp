#include <cuda_median_filter/detail/image_filter_config.h>

#include <gmock/gmock.h>

#include <cstdint>

namespace quxflux::detail
{
  TEST(kernel_configuration, get_block_bounds)
  {
    EXPECT_EQ(image_filter_config::get_block_bounds(16, 1), (bounds{16 * 1, 16}));
    EXPECT_EQ(image_filter_config::get_block_bounds(16, 4), (bounds{16 * 4, 16}));
  }

  TEST(kernel_configuration, get_block_bounds_with_apron)
  {
    EXPECT_EQ(image_filter_config::get_block_bounds_with_apron(16, 5, 1), (bounds{16 * 1 + 4, 16 + 4}));
    EXPECT_EQ(image_filter_config::get_block_bounds_with_apron(16, 5, 4), (bounds{16 * 4 + 4, 16 + 4}));
  }

  TEST(kernel_configuration, get_shared_buf_row_pitch)
  {
    EXPECT_EQ(image_filter_config::get_shared_buf_row_pitch<uint8_t>(16, 5, 1), 16 + 4);
    EXPECT_EQ(image_filter_config::get_shared_buf_row_pitch<uint8_t>(16, 5, 4), (16 * 4) + 4);
    EXPECT_EQ(image_filter_config::get_shared_buf_row_pitch<uint8_t>(16, 7, 4), 72);
    EXPECT_EQ(image_filter_config::get_shared_buf_row_pitch<uint32_t>(16, 7, 1), (16 + 6) * 4);
  }

  TEST(kernel_configuration, get_shared_buf_size)
  {
    EXPECT_EQ(image_filter_config::get_shared_buf_row_pitch<uint8_t>(16, 7, 1), 22);
    EXPECT_EQ(image_filter_config::get_shared_buf_row_pitch<uint8_t>(16, 7, 4), 72);
    EXPECT_EQ(image_filter_config::get_shared_buf_row_pitch<uint32_t>(16, 7, 1), 88);
  }
}  // namespace quxflux::detail
