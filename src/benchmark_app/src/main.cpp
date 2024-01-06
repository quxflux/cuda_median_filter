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

#include <filter_database.h>
#include <filter_spec_factory.h>

#include <shared/cuda/util.h>
#include <shared/image.h>
#include <shared/fill_image_random.h>
#include <shared/ppm.h>
#include <shared/statistics.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace
{
  using namespace quxflux;

  bool operator<(const filter_configuration& first, const filter_configuration& second)
  {
    constexpr auto tie = [](const auto& config) {
      return std::tie(config.filter_size, config.library_name, config.filter_size, config.variant);
    };

    return tie(first) < tie(second);
  }

  auto get_filter_impls()
  {
    auto impls = get_registered_filters();

    std::sort(std::begin(impls), std::end(impls), [](const auto& first_impl, const auto& second_impl) {
      return std::get<0>(first_impl) < std::get<0>(second_impl);
    });

    return impls;
  }

  auto csv_file_name_for_filter_configuration(const filter_configuration& filter_config)
  {
    std::stringstream ss;
    ss << filter_config << ".csv";
    return ss.str();
  }
}  // namespace

int main(int argc, char** args)
{
  bool dump_images = false;

  for (int i = 1; i < argc; ++i)
  {
    if (std::string_view(args[i]) == "--dump")
      dump_images = true;
  }

  for (const auto& filter_config_and_impl : get_filter_impls())
  {
    for_each_filter_value_type([&]([[maybe_unused]] auto value) {
      using value_type = decltype(value);

      const auto& [filter_config, filter_impl] = filter_config_and_impl;

      if (std::type_index{typeid(value_type)} != filter_config.value_type)
        return;

      std::ofstream ofs(csv_file_name_for_filter_configuration(filter_config));
      ofs << std::fixed << "image_size (megapixels)\texec_time (ms)\tprocessing_speed (MP/s)\n";

      std::cout << filter_config << std::endl;

      for (std::int32_t image_size = 1000; image_size <= 5000; image_size += 1000)
      {
        const auto num_megapixels = (image_size * image_size) / 1'000'000;

        std::cout << std::fixed << std::setprecision(2) << num_megapixels << " MP" << std::endl;

        const bounds<std::int32_t> bounds{image_size, image_size};

        auto src_image = make_host_pinned_image<value_type>(bounds);
        auto dst_image = make_host_pinned_image<value_type>(bounds);

        fill_image_random(src_image);

        std::any any_result{dst_image};

        // first run to warm up kernels
        filter_impl->filter(src_image, any_result);

        constexpr std::size_t num_runs = 100;

        std::array<double, num_runs> measurements{};

        for (std::size_t i = 0; i < num_runs; ++i)
        {
          std::cout << "[" << i + 1 << " / " << num_runs << "]: ";

          using clock = std::chrono::high_resolution_clock;

          const auto start = clock::now();
          filter_impl->filter(src_image, any_result);
          const auto stop = clock::now();

          if (dump_images && i == 0)
          {
            std::stringstream ss;
            ss << "./" << num_megapixels << "_" << filter_config << ".ppm";

            export_grayscale_ppm(dst_image, ss.str());
          }

          const auto num_milliseconds =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(stop - start).count();
          const auto megapixels_per_second = static_cast<double>(num_megapixels) / (num_milliseconds / 1000);

          ofs << num_megapixels << '\t' << num_milliseconds << '\t' << megapixels_per_second << '\n';

          std::cout << megapixels_per_second << " MP/s";

          if (i + 1 < num_runs)
            std::cout << '\r';

          measurements[i] = megapixels_per_second;
        }

        const auto [processing_speed_mean, processing_speed_stddev] = calculate_mean_and_standard_deviation(
          measurements.begin(), measurements.end());

        std::cout << " mean: " << processing_speed_mean << ", stddev: " << processing_speed_stddev << std::endl;
      }

      std::cout << std::endl << std::endl;
    });
  }

  return 0;
}
