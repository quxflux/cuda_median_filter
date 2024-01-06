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

#pragma once

#include <shared/image.h>

#include <cuda_median_filter/detail/primitives.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace quxflux
{
  [[nodiscard]] inline image<std::uint8_t> import_grayscale_ppm(const std::filesystem::path& path)
  {
    std::ifstream ifs(path, std::ios_base::binary);

    if (!ifs)
      throw std::runtime_error("Cannot open file at path " + path.string());

    std::string ppm_marker;
    ifs >> ppm_marker;

    if (ppm_marker != "P5")
      throw std::runtime_error("Unsupported PPM file format (was expecting P5 header)");

    ifs.ignore(1);

    char c;
    ifs.get(c);
    if (c == '#')
    {
      while (c != '\n')
        ifs.get(c);
    } else
      ifs.putback(c);

    std::int32_t width = 0;
    std::int32_t height = 0;
    std::int32_t max_val = 0;

    ifs >> width >> height >> max_val;

    if (max_val != 255)
      throw std::runtime_error("Unsupported PPM file format");

    ifs.ignore(1);

    image<std::uint8_t> image = make_host_image<std::uint8_t>({height, width});

    for (int i = 0; i < height; ++i)
      ifs.read(reinterpret_cast<char*>(image.row_data_ptr(i)), width);

    return image;
  }

  inline void export_grayscale_ppm(const image<std::uint8_t>& image, const std::filesystem::path& path)
  {
    std::ofstream ofs(path, std::ios_base::binary);

    if (!ofs)
      throw std::runtime_error("Cannot open file at path " + path.string() + " for writing");

    const auto width = image.bounds().width;
    const auto height = image.bounds().height;
    constexpr auto max_val = std::numeric_limits<std::uint8_t>::max();

    ofs << "P5" << '\n';
    ofs << width << '\n';
    ofs << height << '\n';
    ofs << static_cast<std::uint32_t>(max_val) << '\n';

    for (int i = 0; i < height; ++i)
      ofs.write(reinterpret_cast<const char*>(image.row_data_ptr(i)), width);
  }
}  // namespace quxflux
