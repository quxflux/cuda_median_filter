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

#include <abstract_filter_impl.h>

#include <cuda_median_filter/cuda_median_filter.h>

#include <filter_database.h>
#include <filter_spec_factory.h>

#include <shared/cuda/gpu_image.h>
#include <shared/cuda/stream_handle.h>
#include <shared/image_transfer.h>

namespace quxflux
{
  namespace
  {
    template<typename FilterSpec>
    struct filter_impl : abstract_filter_impl
    {
      using T = typename FilterSpec::value_type;

      static filter_configuration filter_configuration()
      {
        ::quxflux::filter_configuration configuration{"cuda_median_filter", typeid(typename FilterSpec::value_type),
                                                      FilterSpec::filter_size};

        return configuration;
      };

      void filter(const std::any& source_image, std::any& target_image) override
      {
        const auto& source = std::any_cast<const image<T>&>(source_image);
        auto& target = std::any_cast<image<T>&>(target_image);

        const auto bounds = source.bounds();

        if (gpu_img_source.bounds() != bounds)
        {
          gpu_img_source = gpu_image<T>(bounds);
          gpu_img_target = gpu_image<T>(bounds);
        }

        {
          stream_handle stream;

          transfer(source, gpu_img_source, stream);

          median_2d_async<T, FilterSpec::filter_size>(gpu_img_source.data(), gpu_img_source.row_pitch_in_bytes(),
                                                      gpu_img_target.data(), gpu_img_target.row_pitch_in_bytes(),
                                                      bounds.width, bounds.height, stream);

          transfer(gpu_img_target, target, stream);

          stream.synchronize();
        }
      }

      gpu_image<T> gpu_img_source;
      gpu_image<T> gpu_img_target;
    };

    template<typename FilterSpec>
    void register_filter_impl()
    {
      using impl_t = filter_impl<FilterSpec>;
      register_filter({impl_t::filter_configuration(), std::make_shared<impl_t>()});
    }

    auto filter_registrar = [] {
      for_each_filter_spec([&](auto filter_spec) {
        using filter_spec_t = decltype(filter_spec);

        register_filter_impl<filter_spec_t>();
      });

      return 0;
    }();
  }  // namespace
}  // namespace quxflux
