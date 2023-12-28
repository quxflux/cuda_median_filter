#pragma once

#include <cuda_median_filter/detail/cuda/wrap_cuda.h>
#include <cuda_median_filter/detail/image_source_target.h>

namespace quxflux
{
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
}
