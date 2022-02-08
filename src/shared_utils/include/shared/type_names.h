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
#include <string_view>
#include <typeindex>
#include <typeinfo>

namespace quxflux
{
  inline std::string_view to_string(const std::type_index& type_index)
  {
    if (type_index == typeid(std::uint8_t))
      return "uint8";

    if (type_index == typeid(std::uint16_t))
      return "uint16";

    if (type_index == typeid(std::uint32_t))
      return "uint32";

    if (type_index == typeid(std::uint64_t))
      return "uint64";

    if (type_index == typeid(float))
      return "float";

    return type_index.name();
  }
}
