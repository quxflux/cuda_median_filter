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

#include <sycl/sycl.hpp>

#include <gmock/gmock.h>

#include <numeric>
#include <ranges>

namespace quxflux
{
  TEST(sycl_impl, test_compile)
  {
    sycl::queue queue;

    std::vector<int> input_vector(1024);
    std::iota(input_vector.begin(), input_vector.end(), int{0});
    sycl::buffer input_buf{input_vector};

    std::vector<int> output_vector(1024);
    sycl::buffer output_buf{output_vector};

    queue.submit([&](sycl::handler &handler) {
      sycl::accessor input{input_buf, sycl::read_only};
      sycl::accessor output{output_buf, sycl::write_only};

      handler.require(input);
      handler.require(output);

      handler.parallel_for(sycl::range{input.size()}, [=](const sycl::id<1> idx) { output[idx] = input[idx] + 10; });
    });

    output_buf.get_host_access();
    EXPECT_THAT(output_vector, testing::ElementsAreArray(std::views::iota(int{10}, int{10 + 1024})));
  }
}  // namespace quxflux
