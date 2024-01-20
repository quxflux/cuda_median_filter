#pragma once

#include <cuda_median_filter/detail/primitives.h>

#include <span>
#include <stdexcept>

namespace quxflux
{
  template<typename T, typename Byte, typename = std::enable_if_t<std::is_const_v<Byte> == std::is_const_v<T>>>
  std::span<T> reinterpret_as(const std::span<Byte> bytes)
  {
    if (bytes.size() % sizeof(T) != 0)
      throw std::invalid_argument("Invalid number of bytes");

    return {reinterpret_cast<T*>(bytes.data()), bytes.size() / sizeof(T)};
  }
}  // namespace quxflux
