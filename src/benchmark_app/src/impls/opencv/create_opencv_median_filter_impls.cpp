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

#include <impls/opencv/cv_mat_image_view.h>

#include <filter_spec_factory.h>
#include <filter_database.h>

#include <shared/image.h>

namespace quxflux
{
  namespace
  {
    template<typename FilterSpec>
    struct filter_impl : abstract_filter_impl
    {
      static ::quxflux::filter_configuration filter_configuration()
      {
        return {"opencv", typeid(typename FilterSpec::value_type), FilterSpec::filter_size, ""};
      };

      void filter(const std::any& source_image, std::any& target_image) override
      {
        using T = typename FilterSpec::value_type;

        const auto& source = std::any_cast<const image<T>&>(source_image);
        auto& target = std::any_cast<image<T>&>(target_image);

        const auto cv_source = create_mat_view_for_image(source);
        const auto cv_target = create_mat_view_for_image(target);

        cv::medianBlur(cv_source, cv_target, FilterSpec::filter_size);
      }
    };

    [[maybe_unused]] auto filter_registrar = [] {
      for_each_filter_spec([&](auto filter_spec) {
        using filter_spec_t = decltype(filter_spec);

        if constexpr (cv_mat_supports_type<typename filter_spec_t::value_type>::value)
        {
          using impl_t = filter_impl<filter_spec_t>;
          register_filter({impl_t::filter_configuration(), std::make_shared<impl_t>()});
        }
      });

      return 0;
    }();
  }  // namespace
}  // namespace quxflux
