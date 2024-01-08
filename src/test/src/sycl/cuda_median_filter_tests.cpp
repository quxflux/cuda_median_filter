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

#include <cuda_median_filter/cuda_median_filter.h>
#include <cuda_median_filter/detail/primitives.h>

#include <util/image_matcher.h>
#include <util/naive_median_filter_impl.h>

#include <shared/fill_image_random.h>
#include <shared/image.h>

#include <gmock/gmock.h>

namespace quxflux
{
  TEST(sycl_impl, test_compile)
  {
    using T = std::uint8_t;
    static constexpr size_t filter_size = 5;
    constexpr auto bounds = ::quxflux::bounds<std::int32_t>{128, 256};

    image<T> input_image = make_host_image<T>(bounds);
    fill_image_random(input_image);

    image<T> expected = make_host_image<T>(bounds);
    filter_image<filter_size, T>(input_image, expected);

    image<T> output_image = make_host_image<T>(bounds);

    const sycl::range<2> sycl_image_range = {static_cast<size_t>(bounds.height),
                                             input_image.row_pitch_in_bytes() / sizeof(T)};

    sycl::buffer<T, 2> sycl_input_buf{input_image.data_ptr(), sycl_image_range};
    sycl::buffer<T, 2> sycl_output_buf{output_image.data_ptr(), sycl_image_range};

    sycl::queue queue;
    median_2d_async<filter_size>(sycl_input_buf, sycl_output_buf, queue, bounds.width);

    sycl_output_buf.get_host_access();

    EXPECT_THAT(output_image, is_equal_to_image(expected));
  }
}  // namespace quxflux
