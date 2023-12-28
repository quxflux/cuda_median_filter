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

#include <gtest/gtest.h>

#include <cuda_median_filter/cuda_median_filter.h>

#include <util/filter_types_factory.h>
#include <util/image_matcher.h>
#include <util/naive_median_filter_impl.h>

#include <shared/fill_image_random.h>
#include <shared/cuda/gpu_image.h>
#include <shared/image_transfer.h>
#include <shared/cuda/stream_handle.h>
#include <shared/cuda/texture_handle.h>
#include <shared/type_names.h>

#include <array>
#include <string>
#include <type_traits>

namespace quxflux
{
  namespace
  {
    using types_to_test =
      rewrap_list<::testing::Types,
                  generate_all_filter_specs<metal::list<std::uint8_t, std::uint16_t, std::uint32_t, float>,
                                            metal::numbers<1, 3, 5, 7>>>;

    struct test_type_name_generator
    {
      template<typename T>
      static std::string GetName(int)
      {
        const auto type_name = std::string{to_string(typeid(typename T::value_type))};
        return type_name + "_" + std::to_string(T::filter_size);
      }
    };

    template<typename FilterSpec>
    struct filter_impl_test : ::testing::Test
    {};

    namespace image_source_type
    {
      struct pitched_array_2d
      {};

      struct texture
      {};
    }  // namespace image_source_type

    template<typename FilterSpec, typename ImageSourceType>
    void run_gpu_cpu_equality_test(const ImageSourceType&)
    {
      using T = typename FilterSpec::value_type;

      constexpr auto bounds = ::quxflux::bounds<std::int32_t>{128, 256};

      image<T> cpu_buf(bounds);
      image<T> expected(bounds);

      gpu_image<T> gpu_img_src(bounds);
      gpu_image<T> gpu_img_result(bounds);

      fill_image_random(cpu_buf);
      filter_image<FilterSpec::filter_size, T>(cpu_buf, expected);

      {
        stream_handle stream;
        transfer(cpu_buf, gpu_img_src, stream);

        if constexpr (std::is_same_v<ImageSourceType, image_source_type::texture>)
        {
          const texture_handle<T> tex(gpu_img_src.data(), gpu_img_src.bounds(), gpu_img_src.row_pitch_in_bytes());

          median_2d_async<T, FilterSpec::filter_size>(tex, gpu_img_result.data(), gpu_img_result.row_pitch_in_bytes(),
                                                      bounds.width, bounds.height, stream);

        } else if constexpr (std::is_same_v<ImageSourceType, image_source_type::pitched_array_2d>)
        {
          median_2d_async<T, FilterSpec::filter_size>(gpu_img_src.data(), gpu_img_src.row_pitch_in_bytes(),
                                                      gpu_img_result.data(), gpu_img_result.row_pitch_in_bytes(),
                                                      bounds.width, bounds.height, stream);
        }

        transfer(gpu_img_result, cpu_buf, stream);
      }

      EXPECT_THAT(cpu_buf, is_equal_to_image(expected));
    }

  }  // namespace

  TYPED_TEST_SUITE(filter_impl_test, types_to_test, test_type_name_generator);

  TYPED_TEST(filter_impl_test, call_with_empty_image_does_not_fail)
  {
    using T = typename TypeParam::value_type;

    constexpr bounds<std::int32_t> empty_bounds{};

    gpu_image<T> gpu_img_src(empty_bounds);
    gpu_image<T> gpu_img_dst(empty_bounds);

    EXPECT_NO_THROW((median_2d_async<T, TypeParam::filter_size>(
      gpu_img_src.data(), gpu_img_src.row_pitch_in_bytes(), gpu_img_dst.data(), gpu_img_dst.row_pitch_in_bytes(),
      gpu_img_src.bounds().width, gpu_img_src.bounds().height)));
  }

  TYPED_TEST(filter_impl_test, gpu_result_equals_naive_cpu_implementation_when_using_pitched_image_source)
  {
    run_gpu_cpu_equality_test<TypeParam>(image_source_type::pitched_array_2d{});
  }

  TYPED_TEST(filter_impl_test, gpu_result_equals_naive_cpu_implementation_when_using_texture_image_source)
  {
    run_gpu_cpu_equality_test<TypeParam>(image_source_type::texture{});
  }
}  // namespace quxflux
