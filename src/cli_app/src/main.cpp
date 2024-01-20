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

#include <shared/image.h>
#include <shared/ppm.h>

#include <lyra/lyra.hpp>

#include <filesystem>
#include <iostream>
#include <ranges>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <string_view>

using namespace quxflux;

namespace
{
  enum class implementation
  {
    cuda,
    sycl
  };

  constexpr auto available_implementations = std::to_array<std::pair<implementation, std::string_view>>({
    {implementation::cuda, "cuda"},
#ifdef SYCL_LANGUAGE_VERSION
    {implementation::sycl, "sycl"},
#endif
  });

  constexpr std::optional<implementation> implementation_from_string(std::string_view s)
  {
    const auto it = std::ranges::find(available_implementations, s,
                                      &std::pair<implementation, std::string_view>::second);
    if (it == available_implementations.end())
      return std::nullopt;
    return {it->first};
  }
}  // namespace

void filter_image_cuda(const image<std::uint8_t>& input, image<std::uint8_t>& output);
void filter_image_sycl(const image<std::uint8_t>& input, image<std::uint8_t>& output);

int main(int argc, char** args)
{
  std::filesystem::path input_file;
  size_t num_runs = 10;
  bool show_help = false;
  std::string chosen_implementation = std::string{available_implementations.front().second};

  auto cli = lyra::cli()              //
             | lyra::help(show_help)  //
             | lyra::opt(num_runs, "num_runs")("Number of runs to perform");

  if constexpr (available_implementations.size() > 1)
  {
    std::ostringstream oss;
    for (const auto impl :
         available_implementations | std::views::values | std::views::take(available_implementations.size() - 1))
      oss << impl << "|";
    oss << available_implementations.back().second;

    cli |= lyra::opt(chosen_implementation, oss.str())["-i"]["--impl"]("Implementation to use");
  }

  cli |= lyra::arg(input_file, "input")("Path to input file (pgm format)");

  if (const auto result = cli.parse({argc, args}); !result)
  {
    std::cerr << "Error in command line: " << result.message() << std::endl;
    return EXIT_FAILURE;
  }

  if (show_help)
  {
    std::cout << cli;
    return EXIT_SUCCESS;
  }

  if (input_file.extension() != ".pgm")
  {
    std::cerr << "only pgm files are supported" << std::endl;
    return EXIT_FAILURE;
  }

  implementation impl;

  if (const auto converted = implementation_from_string(chosen_implementation); !converted)
  {
    std::cerr << "unknown implementation: " << chosen_implementation << std::endl;
    return EXIT_FAILURE;
  } else
  {
    impl = converted.value();
  }

  image<std::uint8_t> input_img;

  try
  {
    input_img = import_pgm(input_file.string());
  }
  catch (const std::exception& e)
  {
    std::cerr << "import of file at path " << input_file << " failed:\n" << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  const auto bounds = input_img.bounds();
  image<std::uint8_t> output_img = make_host_image<std::uint8_t>(bounds);

  const auto num_megapixels = (bounds.width * bounds.height) / 1'000'000;

  for (std::size_t i = 0; i < num_runs; ++i)
  {
    using clock = std::chrono::high_resolution_clock;
    const auto start = clock::now();

    switch (impl)
    {
      case implementation::cuda:
        filter_image_cuda(input_img, output_img);
        break;
      case implementation::sycl:
#ifdef SYCL_LANGUAGE_VERSION
        filter_image_sycl(input_img, output_img);
#endif
        break;
    }

    const auto stop = clock::now();
    const auto megapixels_per_second = static_cast<double>(num_megapixels) /
                                       std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

    std::cout << megapixels_per_second << " MP/s\n";
  }

  export_pgm(output_img, input_file.parent_path() / input_file.stem() += "_filtered.pgm");

  return EXIT_SUCCESS;
}
