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

#include <type_traits>

#include <metal.hpp>

namespace quxflux
{
  // clang-format: off
  // Using custom tags instead of std::true_type/std::false_type to improve readability in test explorer
  struct use_vectorization {};
  struct no_vectorization {};
  // clang-format on

  template<typename DataType, typename FilterSize, typename Vectorize>
  struct filter_spec
  {
    using value_type = DataType;
    static constexpr auto filter_size = FilterSize::value;
    static constexpr auto vectorize = std::is_same_v<Vectorize, use_vectorization>;
  };

  namespace detail
  {
    template<typename DataType, typename FilterSizes>
    auto generate_all_filter_specs_impl() -> metal::transform<
      metal::partial<metal::lambda<metal::apply>, metal::lambda<filter_spec>>,
      metal::cartesian<metal::list<DataType>, FilterSizes, metal::list<no_vectorization, use_vectorization>>>
    {
      return {};
    }

    // Utility function to rewrap a metal::list into another variadic template.
    // This is used for transforming metal::lists into testing::Types for GoogleTest.
    template<template<typename...> typename VariadicT>
    struct rewrap_list_impl
    {
      template<typename... ListItems>
      auto operator()(const metal::list<ListItems...>* const) -> VariadicT<ListItems...>*
      {
        return {};
      }
    };
  }  // namespace detail

  template<typename DataType, typename FilterSizes>
  using generate_all_filter_specs = decltype(detail::generate_all_filter_specs_impl<DataType, FilterSizes>());

  template<template<typename...> typename VariadicTemplate, typename List>
  using rewrap_list =
    std::remove_pointer_t<decltype(detail::rewrap_list_impl<VariadicTemplate>{}(static_cast<const List*>(nullptr)))>;
}  // namespace quxflux
