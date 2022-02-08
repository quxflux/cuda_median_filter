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

#include <algorithm>
#include <numeric>

namespace quxflux
{
  template<typename It>
  constexpr auto calculate_mean_and_standard_deviation(const It begin, const It end)
  {
    using T = typename std::iterator_traits<It>::value_type;

    const T n = static_cast<T>(std::distance(begin, end));

    const T mean = std::accumulate(begin, end, T{0}) / n;

    const T sum_of_squared_differences = std::accumulate(begin, end, T{0}, [=](const T acc, const T v) {
      const T diff = v - mean;
      return acc + diff * diff;
    });

    struct result
    {
      T mean;
      T stddev;
    };

    return result{mean, std::sqrt(sum_of_squared_differences / n)};
  }

  template<typename It>
  constexpr auto calculate_percentile(const It begin, const It end, const float percentile)
  {
    const std::size_t n = std::distance(begin, end);

    const std::size_t offset = static_cast<std::size_t>((n - 1) * percentile);

    if (offset >= n)
      throw std::runtime_error("Invalid percentile");

    const It nth = begin + offset;

    std::nth_element(begin, nth, end);

    return *nth;
  }
}  // namespace quxflux
