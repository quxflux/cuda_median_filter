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
      static constexpr int NW = -1;

      std::array<int, 4 * 4> output_pixel_values = [] {
        std::array<int, 4 * 4> result{};
        std::ranges::fill(result, NW);
        return result;
      }();

      pitched_array_image_source<int> src{std::as_bytes(std::span{input_pixel_values}).data(), {6, 6}, 6 * sizeof(int)};
      pitched_array_image_target<int> dst{
        std::as_writable_bytes(std::span{output_pixel_values}).data(), {4, 4}, 4 * sizeof(int)};
    };
  }  // namespace

  TEST_F(load_neighbor_pixels_test, dst_holds_expected_data_for_single_thread)
  {
    load_neighbor_pixels<int, load_neighbor_params{.block_size = 2, .filter_size = 3, .local_bounds = {2, 2}}>(
      dst, src, {0, 0}, {1, 1});

    constexpr std::array<int, 4 * 4> expected_pixel_values{
      NW, NW, NW, NW,  //
      NW, 1,  NW, 3,   //
      NW, NW, NW, NW,  //
      NW, 13, NW, 15,  //
    };

    EXPECT_THAT(output_pixel_values, ::testing::ElementsAreArray(expected_pixel_values));
  }

  TEST_F(load_neighbor_pixels_test, dst_holds_expected_data_for_single_thread_when_apron_is_offset)
  {
     load_neighbor_pixels<int, load_neighbor_params{.block_size = 2, .filter_size = 3, .local_bounds = {2, 2}}>(
      dst, src, {1, 0}, {0, 0});

    constexpr std::array<int, 4 * 4> expected_pixel_values{
      0,  NW, 0,  NW,  //
      NW, NW, NW, NW,  //
      8,  NW, 10, NW,  //
      NW, NW, NW, NW,  //
    };

    EXPECT_THAT(output_pixel_values, ::testing::ElementsAreArray(expected_pixel_values));
  }

  TEST_F(load_neighbor_pixels_test, dst_holds_expected_data_for_all_block_threads_when_at_border)
  {
    constexpr point<std::int32_t> apron_offset{0, 0};

    for (std::int32_t y = 0; y < 4; ++y)
      for (std::int32_t x = 0; x < 4; ++x)
        load_neighbor_pixels<int, load_neighbor_params{.block_size = 2, .filter_size = 3, .local_bounds = {2, 2}}>(
          dst, src, apron_offset, {x, y});

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
    constexpr point<std::int32_t> apron_offset{1, 1};

    for (std::int32_t y = 0; y < 4; ++y)
      for (std::int32_t x = 0; x < 4; ++x)
        load_neighbor_pixels<int, load_neighbor_params{.block_size = 2, .filter_size = 3, .local_bounds = {2, 2}}>(
          dst, src, apron_offset, {x, y});

    constexpr std::array<int, 4 * 4> expected_pixel_values{
      8,  9,  10, 11,  //
      14, 15, 16, 17,  //
      20, 21, 22, 23,  //
      26, 27, 28, 29,  //
    };

    EXPECT_THAT(output_pixel_values, ::testing::ElementsAreArray(expected_pixel_values));
  }
}  // namespace quxflux::detail
