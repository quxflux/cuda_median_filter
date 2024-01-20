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

#include <shared/image.h>
#include <shared/reinterpret_as.h>

#include <sycl/sycl.hpp>

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
        ::quxflux::filter_configuration configuration{"cuda_median_filter_sycl", typeid(typename FilterSpec::value_type),
                                                      FilterSpec::filter_size};

        return configuration;
      };

      void filter(const std::any& source_image, std::any& target_image) override
      {
        const auto& source = std::any_cast<const image<T>&>(source_image);
        auto& target = std::any_cast<image<T>&>(target_image);

        const auto bounds = source.bounds();

        {
          sycl::queue queue;

          const sycl::range<2> sycl_image_range = {static_cast<size_t>(bounds.height),
                                                   source.row_pitch_in_bytes() / sizeof(T)};

          sycl::buffer<T, 2> sycl_input_buf{reinterpret_as<const T>(source.data()).data(), sycl_image_range};
          sycl::buffer<T, 2> sycl_output_buf{reinterpret_as<T>(target.data()).data(), sycl_image_range};

          median_2d_async<FilterSpec::filter_size>(sycl_input_buf, sycl_output_buf, queue, bounds.width);

          sycl_output_buf.get_host_access();
        }
      }
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
