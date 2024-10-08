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

#include <cuda_median_filter/detail/cuda_wrap.h>
#include <cuda_median_filter/detail/pitched_array_accessor.h>
#include <cuda_median_filter/detail/primitives.h>

#include <cstring>
#include <type_traits>

namespace quxflux
{
  namespace detail
  {
    template<typename T>
    class bounded_image
    {
    public:
      using value_type = T;

      constexpr bounded_image(const bounds<std::int32_t>& bounds) : bounds_(bounds) {}
      constexpr auto bounds() const { return bounds_; }

    private:
      ::quxflux::bounds<std::int32_t> bounds_;
    };
  }  // namespace detail

  /**
   * @brief pitched_array_image_source allows to (read) access a global 2d array
   */
  template<typename T>
  class pitched_array_image_source : public detail::bounded_image<T>,
                                     public pitched_array_accessor<T, detail::const_access>
  {
  public:
    constexpr pitched_array_image_source(const std::byte* dev_ptr, const ::quxflux::bounds<std::int32_t>& bounds,
                                         const std::int32_t pitch_in_bytes)
      : detail::bounded_image<T>(bounds), pitched_array_accessor<T, detail::const_access>(dev_ptr, pitch_in_bytes)
    {}
  };

  /**
   * @brief Texture image_source allows to (read) access a texture
   */
  template<typename T>
  class texture_image_source : public detail::bounded_image<T>
  {
  public:
    constexpr texture_image_source(const cudaTextureObject_t texture, const ::quxflux::bounds<std::int32_t>& bounds)
      : detail::bounded_image<T>(bounds), texture_(texture)
    {}

    constexpr T get(const point<std::int32_t>& coord) const { return tex2D<T>(texture_, coord.x, coord.y); }

  private:
    cudaTextureObject_t texture_;
  };

  /**
   * @brief pitched_array_image_target allows to (write) access a global 2d array
   */
  template<typename T>
  class pitched_array_image_target : public detail::bounded_image<T>,
                                     public pitched_array_accessor<T, detail::mutable_access>
  {
  public:
    constexpr pitched_array_image_target(std::byte* dev_ptr, const ::quxflux::bounds<std::int32_t>& bounds,
                                         const std::int32_t pitch_in_bytes)
      : detail::bounded_image<T>(bounds), pitched_array_accessor<T, detail::mutable_access>(dev_ptr, pitch_in_bytes)
    {}
  };
}  // namespace quxflux
