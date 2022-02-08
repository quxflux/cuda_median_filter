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

#if defined(__GNUC__) || defined(__clang__)
_Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wsign-conversion\"")
#endif

#include <metal/list.hpp>

#if defined(__GNUC__) || defined(__clang__)
  _Pragma("GCC diagnostic pop")
#endif

    namespace quxflux
{
  using filter_value_types = metal::list<std::uint8_t /*, std::uint16_t, std::uint32_t, float*/>;
  using filter_sizes = metal::numbers<3, 5, 7, 9, 11, 13, 15>;

  template<typename ValueType, typename FilterSize>
  struct filter_spec
  {
    using value_type = ValueType;
    static inline constexpr std::int32_t filter_size = FilterSize::value;
  };

  namespace detail
  {
    template<typename DataTypes, typename FilterSizes>
    auto generate_filter_specs()
      -> metal::transform<metal::partial<metal::lambda<metal::apply>, metal::lambda<filter_spec>>,
                          metal::cartesian<DataTypes, FilterSizes>>
    {
      return {};
    }

    struct for_each_list_item_impl
    {
      template<typename F, typename... Specs>
      constexpr void operator()(F&& f, metal::list<Specs...>) const
      {
        (f(Specs{}), ...);
      }
    };

  }  // namespace detail

  template<typename F>
  void for_each_filter_spec(F && f)
  {
    detail::for_each_list_item_impl{}(f, decltype(detail::generate_filter_specs<filter_value_types, filter_sizes>()){});
  }

  template<typename F>
  void for_each_filter_value_type(F && f)
  {
    detail::for_each_list_item_impl{}(f, filter_value_types{});
  }

}  // namespace quxflux
