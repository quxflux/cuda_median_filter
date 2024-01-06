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

#include <shared/cuda/util.h>
#include <shared/cuda/gpu_image.h>
#include <shared/cuda/stream_handle.h>
#include <shared/image_transfer.h>
#include <shared/image_transfer.h>
#include <shared/ppm.h>
#include <shared/fill_image_random.h>

#include <filesystem>
#include <iostream>
#include <stdlib.h>

using namespace quxflux;

int main(int argc, char** args)
{
  if (argc <= 1)
  {
    std::cerr << "input filepath argument missing" << std::endl;
    return EXIT_FAILURE;
  }

  const auto file_path = std::filesystem::path(args[1]);

  if (file_path.extension() != ".ppm")
  {
    std::cerr << "only ppm files are supported" << std::endl;
    return EXIT_FAILURE;
  }

  using T = std::uint8_t;

  image<T> input_img;

  try
  {
    input_img = import_grayscale_ppm(file_path.string());
  }
  catch (const std::exception& e)
  {
    std::cerr << "import of file at path " << file_path << " failed:\n" << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  const auto bounds = input_img.bounds();
  image<T> output_img = make_host_image<T>(bounds);

  const auto num_megapixels = (bounds.width * bounds.height) / 1'000'000;
  {
    gpu_image<T> input_img_gpu = make_gpu_image<T>(bounds);
    gpu_image<T> output_img_gpu = make_gpu_image<T>(bounds);

    for (std::size_t i = 0; i < 10; ++i)
    {
      using clock = std::chrono::high_resolution_clock;

      const auto start = clock::now();
      {
        stream_handle stream;
        transfer(input_img, input_img_gpu, stream);
        median_2d_async<T, 7>(input_img_gpu.data_ptr(), input_img_gpu.row_pitch_in_bytes(), output_img_gpu.data_ptr(),
                              output_img_gpu.row_pitch_in_bytes(), bounds.width, bounds.height, stream);
        transfer(output_img_gpu, output_img, stream);
      }

      const auto stop = clock::now();
      const auto num_milliseconds =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(stop - start).count();
      const auto megapixels_per_second = static_cast<double>(num_megapixels) / (num_milliseconds / 1000);

      std::cout << megapixels_per_second << " MP/s\n";
    }
  }

  export_grayscale_ppm(output_img, file_path.parent_path() / file_path.stem() += "_filtered.ppm");

  return EXIT_SUCCESS;
}
