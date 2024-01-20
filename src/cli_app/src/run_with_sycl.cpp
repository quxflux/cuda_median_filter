#include <cuda_median_filter/cuda_median_filter.h>

#include <shared/image.h>
#include <shared/reinterpret_as.h>

#include <cstdint>

using namespace quxflux;

void filter_image_sycl(const image<std::uint8_t>& input, image<std::uint8_t>& output)
{
  if (input.bounds() != output.bounds())
    throw std::invalid_argument("input and output images must have the same bounds");

  const auto bounds = input.bounds();
  const sycl::range<2> sycl_image_range = {static_cast<size_t>(bounds.height),
                                           input.row_pitch_in_bytes() / sizeof(std::uint8_t)};

  sycl::buffer<std::uint8_t, 2> sycl_input_buf{reinterpret_as<const std::uint8_t>(input.data()).data(), sycl_image_range};
  sycl::buffer<std::uint8_t, 2> sycl_output_buf{reinterpret_as<std::uint8_t>(output.data()).data(), sycl_image_range};

  sycl::queue queue;
  median_2d_async<7>(sycl_input_buf, sycl_output_buf, queue, bounds.width);

  sycl_output_buf.get_host_access();
}
