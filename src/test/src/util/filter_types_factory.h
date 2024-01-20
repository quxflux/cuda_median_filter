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

#include <shared/type_names.h>

#include <gmock/gmock.h>
#include <metal.hpp>

#include <string>
#include <tuple>
#include <type_traits>

namespace quxflux
{
  template<typename UseVectorization, typename DataType, typename FilterSize>
  struct filter_spec
  {
    using value_type = DataType;
    static constexpr auto filter_size = FilterSize::value;
    static constexpr auto vectorize = UseVectorization::value;
  };

  namespace detail
  {
    template<typename VectorizationFlags, typename DataTypes, typename FilterSizes>
    constexpr auto generate_all_filter_specs_impl()
      -> metal::transform<metal::partial<metal::lambda<metal::apply>, metal::lambda<filter_spec>>,
                          metal::cartesian<VectorizationFlags, DataTypes, FilterSizes>>
    {
      return {};
    }

    template<template<typename...> typename VariadicT>
    struct rewrap_list_impl
    {
      template<typename... ListItems>
      constexpr auto operator()(const metal::list<ListItems...>) -> VariadicT<ListItems...>
      {
        return {};
      }
    };

    template<template<typename...> typename VariadicTemplate, typename List>
    using rewrap_list = decltype(detail::rewrap_list_impl<VariadicTemplate>{}(List{}));
  }  // namespace detail

  struct generate_filter_test_spec_name
  {
    template<typename T>
    static std::string GetName(int)
    {
      const auto type_name = std::string{to_string(typeid(typename T::value_type))};
      return type_name + "_" + std::to_string(T::filter_size) + (T::vectorize ? "_vectorized" : "");
    }
  };

  struct test_vectorized
  {};

  struct do_not_test_vectorized
  {};

  template<typename Vectorization, typename DataTypes, typename FilterSizes>
  using generate_all_filter_test_specs = detail::rewrap_list<
    ::testing::Types,
    decltype(detail::generate_all_filter_specs_impl<
             std::conditional_t<std::is_same_v<Vectorization, test_vectorized>,
                                metal::list<std::true_type, std::false_type>, metal::list<std::false_type>>,
             DataTypes, FilterSizes>())>;

}  // namespace quxflux
