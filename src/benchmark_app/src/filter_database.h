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

#include <abstract_filter_impl.h>
#include <filter_configuration.h>

#include <tuple>
#include <vector>

namespace quxflux
{
  using filter_impl_ptr = std::shared_ptr<abstract_filter_impl>;

  void register_filter(const std::tuple<filter_configuration, filter_impl_ptr>& filter);
  std::vector<std::tuple<filter_configuration, filter_impl_ptr>> get_registered_filters();
}
