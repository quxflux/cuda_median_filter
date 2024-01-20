#include <cuda_median_filter/cuda_median_filter.h>

#include <shared/cuda/util.h>
#include <shared/cuda/gpu_image.h>
#include <shared/cuda/stream_handle.h>
#include <shared/cuda/image_transfer.h>
#include <shared/image.h>

#include <cstdint>

using namespace quxflux;

void filter_image_cuda(const image<std::uint8_t>& input, image<std::uint8_t>& output)
{
  if (input.bounds() != output.bounds())
    throw std::invalid_argument("input and output images must have the same bounds");

  const auto bounds = input.bounds();

  stream_handle stream;

  gpu_image<std::uint8_t> input_img_gpu = make_gpu_image<std::uint8_t>(bounds);
  gpu_image<std::uint8_t> output_img_gpu = make_gpu_image<std::uint8_t>(bounds);

  transfer(input, input_img_gpu, stream);
  median_2d_async<std::uint8_t, 7>(input_img_gpu.data_ptr(), input_img_gpu.row_pitch_in_bytes(),
                                   output_img_gpu.data_ptr(), output_img_gpu.row_pitch_in_bytes(), bounds.width,
                                   bounds.height, stream);
  transfer(output_img_gpu, output, stream);
}
