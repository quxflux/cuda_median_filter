#include <cuda_median_filter/detail/load_neighbor_pixels.h>
#include <cuda_median_filter/detail/image_source_target.h>
#include <cuda_median_filter/detail/primitives.h>

#include <gmock/gmock.h>

#include <array>
#include <cstdint>
#include <span>

namespace quxflux::detail
{
  namespace
  {
    template<std::int32_t BlockSize, std::int32_t FilterRadius>
    struct config
    {
      static inline constexpr std::int32_t block_size = BlockSize;
      static inline constexpr std::int32_t filter_radius = FilterRadius;

      static inline constexpr std::int32_t num_pixels_x = BlockSize;
      static inline constexpr std::int32_t num_pixels_y = BlockSize;
    };

    struct load_neighbor_pixels_test : testing::Test
    {
      static constexpr std::array<int, 6 * 6> input_pixel_values = {1,  2,  3,  4,  5,  6,   //
                                                                    7,  8,  9,  10, 11, 12,  //
                                                                    13, 14, 15, 16, 17, 18,  //
                                                                    19, 20, 21, 22, 23, 24,  //
                                                                    25, 26, 27, 28, 29, 30,  //
                                                                    31, 32, 33, 34, 35, 36};

      // indicates a non-written pixel
      static constexpr int NA = -1;

      std::array<int, 4 * 4> output_pixel_values = [] {
        std::array<int, 4 * 4> result{};
        std::ranges::fill(result, NA);
        return result;
      }();

      pitched_array_image_source<int> src{std::as_bytes(std::span{input_pixel_values}).data(), {6, 6}, 6 * sizeof(int)};
      pitched_array_image_target<int> dst{
        std::as_writable_bytes(std::span{output_pixel_values}).data(), {4, 4}, 4 * sizeof(int)};
    };
  }  // namespace

  TEST_F(load_neighbor_pixels_test, dst_holds_expected_data_for_single_thread)
  {
    constexpr std::int32_t block_size = 2;
    constexpr std::int32_t filter_radius = 1;
    load_neighbor_pixels<int, config<block_size, filter_radius>>(dst, src, {0, 0}, {1, 1});

    constexpr std::array<int, 4 * 4> expected_pixel_values{
      NA, NA, NA, NA,  //
      NA, 1,  NA, 3,   //
      NA, NA, NA, NA,  //
      NA, 13, NA, 15,  //
    };

    EXPECT_THAT(output_pixel_values, ::testing::ElementsAreArray(expected_pixel_values));
  }

  TEST_F(load_neighbor_pixels_test, dst_holds_expected_data_for_single_thread_when_apron_is_offset)
  {
    constexpr std::int32_t block_size = 2;
    constexpr std::int32_t filter_radius = 1;
    load_neighbor_pixels<int, config<block_size, filter_radius>>(dst, src, {1, 0}, {0, 0});

    constexpr std::array<int, 4 * 4> expected_pixel_values{
      0,  NA, 0,  NA,  //
      NA, NA, NA, NA,  //
      8,  NA, 10, NA,  //
      NA, NA, NA, NA,  //
    };

    EXPECT_THAT(output_pixel_values, ::testing::ElementsAreArray(expected_pixel_values));
  }

  TEST_F(load_neighbor_pixels_test, dst_holds_expected_data_for_all_block_threads_when_at_border)
  {
    constexpr std::int32_t block_size = 2;
    constexpr std::int32_t filter_radius = 1;

    constexpr point<std::int32_t> apron_offset{0, 0};

    static_for_2d<block_size, block_size>([&](const auto idx) {
      load_neighbor_pixels<int, config<block_size, filter_radius>>(dst, src, apron_offset, {idx.x, idx.y});
    });

    constexpr std::array<int, 4 * 4> expected_pixel_values{
      0, 0,  0,  0,   //
      0, 1,  2,  3,   //
      0, 7,  8,  9,   //
      0, 13, 14, 15,  //
    };

    EXPECT_THAT(output_pixel_values, ::testing::ElementsAreArray(expected_pixel_values));
  }

  TEST_F(load_neighbor_pixels_test, dst_holds_expected_data_for_all_block_threads_when_in_inner)
  {
    constexpr std::int32_t block_size = 2;
    constexpr std::int32_t filter_radius = 1;

    constexpr point<std::int32_t> apron_offset{1, 1};

    static_for_2d<block_size, block_size>([&](const auto idx) {
      load_neighbor_pixels<int, config<block_size, filter_radius>>(dst, src, apron_offset, {idx.x, idx.y});
    });

    constexpr std::array<int, 4 * 4> expected_pixel_values{
      8,  9,  10, 11,  //
      14, 15, 16, 17,  //
      20, 21, 22, 23,  //
      26, 27, 28, 29,  //
    };

    EXPECT_THAT(output_pixel_values, ::testing::ElementsAreArray(expected_pixel_values));
  }
}  // namespace quxflux::detail
