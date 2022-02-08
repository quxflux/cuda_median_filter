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

#include <iterator>
#include <array>
#include <type_traits>

namespace quxflux
{
  namespace detail
  {
    template<typename I>
    constexpr I to_sequential(const std::array<I, 2>& index, const std::array<I, 2>& extents)
    {
      return index[1] * extents[0] + index[0];
    }

    template<typename I>
    constexpr std::tuple<I, I> to_multidimensional(I idx, const std::array<I, 2>& extents)
    {
      const I y = idx / extents[0];
      const I x = idx % extents[0];

      return {x, y};
    }

    template<typename I>
    class index2d_iterator
    {
    public:
      static_assert(std::is_integral_v<I>, "I must be an integral type");

      using index2d = std::tuple<I, I>;

      using iterator_category = std::bidirectional_iterator_tag;
      using difference_type = I;
      using value_type = index2d;
      using reference = index2d;
      using pointer = index2d;

      index2d_iterator() = default;
      index2d_iterator(const index2d_iterator&) = default;
      index2d_iterator(index2d_iterator&&) = default;
      index2d_iterator& operator=(const index2d_iterator&) = default;
      index2d_iterator& operator=(index2d_iterator&&) = default;

      constexpr auto operator==(const index2d_iterator& rhs) const { return current_ == rhs.current_; }
      constexpr auto operator!=(const index2d_iterator& rhs) const { return !(*this == rhs); }

      constexpr index2d_iterator(const std::array<I, 2>& extents) : extents_(extents) {}
      constexpr index2d_iterator(const std::array<I, 2>& index, const std::array<I, 2>& extents)
        : extents_({extents[0], extents[1]}), current_(to_sequential(index, extents))
      {}

      constexpr value_type operator*() const
      {
        return to_multidimensional(current_, std::array{std::get<0>(extents_), std::get<1>(extents_)});
      }

      constexpr auto& operator++()
      {
        current_++;
        return *this;
      }

      constexpr auto operator++(int)
      {
        auto tmp = *this;
        ++(*this);
        return tmp;
      }

      constexpr auto& operator--()
      {
        current_--;
        return *this;
      }

      constexpr auto operator--(int)
      {
        auto tmp = *this;
        --(*this);
        return tmp;
      }

    private:
      index2d extents_{0};
      I current_{0};
    };

    struct index2d_range_impl
    {
      template<typename T>
      constexpr auto operator()(const std::array<T, 2>& extents) const
      {
        struct r
        {
          index2d_iterator<T> begin_;
          index2d_iterator<T> end_;

          constexpr auto begin() const { return begin_; }
          constexpr auto end() const { return end_; }
        };

        return r{index2d_iterator<T>{{0}, extents}, index2d_iterator<T>{{extents[0], extents[1] - 1}, extents}};
      }
    };
  }  // namespace detail

  constexpr inline detail::index2d_range_impl index2d_range;
}  // namespace quxflux
