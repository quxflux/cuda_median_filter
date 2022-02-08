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

#include <cstdint>
#include <ostream>
#include <string_view>
#include <typeindex>
#include <typeinfo>

#include <shared/type_names.h>

namespace quxflux
{
  struct filter_configuration
  {
    std::string_view library_name;
    std::type_index value_type;
    std::int32_t filter_size;
    std::string_view variant;

    constexpr bool operator==(const filter_configuration& rhs) const
    {
      return library_name == rhs.library_name && value_type == rhs.value_type && filter_size == rhs.filter_size &&
             variant == rhs.variant;
    }
  };
}  // namespace quxflux

inline std::ostream& operator<<(std::ostream& os, const quxflux::filter_configuration& config)
{
  os << config.filter_size << "x" << config.filter_size << "_" << quxflux::to_string(config.value_type)
     << "_" << config.library_name;

  if (!config.variant.empty())
    os << "_" << config.variant;

  return os;
}
