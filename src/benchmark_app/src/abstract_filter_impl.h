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

#include <filter_configuration.h>

#include <any>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

namespace quxflux
{
  struct abstract_filter_impl
  {
    abstract_filter_impl() = default;
    abstract_filter_impl(const abstract_filter_impl&) = default;
    abstract_filter_impl(abstract_filter_impl&&) = default;
    abstract_filter_impl& operator=(const abstract_filter_impl&) = default;
    abstract_filter_impl& operator=(abstract_filter_impl&&) = default;

    virtual void filter(const std::any& source_image, std::any& target_image) = 0;

    virtual ~abstract_filter_impl() {}
  };
}  // namespace quxflux
