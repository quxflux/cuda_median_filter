// This file is part of the cuda_median_filter (https://github.com/quxflux/cuda_median_filter).
// Copyright (c) 2024 Lukas Riebel.
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

#include <cuda_median_filter/cuda_median_filter.h>
#include <cuda_median_filter/detail/primitives.h>

#include <util/filter_types_factory.h>
#include <util/image_matcher.h>
#include <util/naive_median_filter_impl.h>

#include <shared/fill_image_random.h>
#include <shared/image.h>

#include <gmock/gmock.h>
#include <metal.hpp>

namespace quxflux
{
  namespace
  {
    using types_to_test =
      rewrap_list<::testing::Types,
                  generate_all_filter_specs<false, metal::list<std::uint8_t, std::uint16_t, std::uint32_t, float>,
                                            metal::numbers<1, 3, 5, 7>>>;

    template<typename FilterSpec>
    struct sycl_filter_impl_test : ::testing::Test
    {};
    TYPED_TEST_SUITE(sycl_filter_impl_test, types_to_test, filter_type_name);

    template<typename T>
    std::span<T> reinterpret_as(const std::span<byte> bytes)
    {
      return {reinterpret_cast<T*>(bytes.data()), bytes.size() / sizeof(T)};
    }
  }  // namespace


  TYPED_TEST(sycl_filter_impl_test, call_with_empty_image_does_not_fail)
  {
    using T = typename TypeParam::value_type;
    sycl::queue queue;

    sycl::buffer<T, 2> gpu_img_src(nullptr, sycl::range<2>{});
    sycl::buffer<T, 2> gpu_img_dst(nullptr, sycl::range<2>{});

    EXPECT_NO_THROW(median_2d_async<TypeParam::filter_size>(gpu_img_src, gpu_img_dst, queue, 0));
  }

  TYPED_TEST(sycl_filter_impl_test, gpu_result_equals_naive_cpu_implementation)
  {
    using T = typename TypeParam::value_type;

    static constexpr auto bounds = ::quxflux::bounds<std::int32_t>{128, 256};

    image<T> input = make_host_image<T>(bounds);
    image<T> expected = make_host_image<T>(bounds);

    fill_image_random(input);
    filter_image<TypeParam::filter_size, T>(input, expected);

    image<T> output = make_host_image<T>(bounds);

    const sycl::range<2> sycl_image_range = {static_cast<size_t>(bounds.height),
                                             input.row_pitch_in_bytes() / sizeof(T)};

    sycl::buffer<T, 2> sycl_input_buf{reinterpret_as<T>(input.data()).data(), sycl_image_range};
    sycl::buffer<T, 2> sycl_output_buf{reinterpret_as<T>(output.data()).data(), sycl_image_range};

    sycl::queue queue;
    median_2d_async<TypeParam::filter_size>(sycl_input_buf, sycl_output_buf, queue, bounds.width);

    sycl_output_buf.get_host_access();

    EXPECT_THAT(output, is_equal_to_image(expected));
  }
}  // namespace quxflux
