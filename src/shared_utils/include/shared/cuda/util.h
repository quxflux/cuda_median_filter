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

#include <cassert>
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>

#include <cuda_median_filter/detail/cuda/wrap_cuda.h>
#include <cuda_median_filter/detail/primitives.h>
#include <shared/image.h>

namespace quxflux
{
  namespace detail
  {
    template<auto Func>
    struct cuda_free_caller
    {
      template<typename T>
      constexpr void operator()(T* device_ptr) const
      {
        [[maybe_unused]] const cudaError_t r = std::invoke(Func, device_ptr);

        assert(r == cudaSuccess && "cudaFree failed!");
      }
    };

    using cuda_free = cuda_free_caller<&cudaFree>;
    using cuda_free_host = cuda_free_caller<&cudaFreeHost>;
  }  // namespace detail

  struct cuda_runtime_error : std::runtime_error
  {
    using std::runtime_error::runtime_error;
  };

  template<typename F, typename... Args>
  void cuda_call(const F& f, Args&&... args)
  {
    const cudaError_t r = f(std::forward<Args>(args)...);

    if (r != cudaSuccess)
      throw cuda_runtime_error(cudaGetErrorString(r));
  }

  template<typename F>
  auto measure_cuda_event_time(const F& f, cudaStream_t stream)
  {
    cudaEvent_t start, stop;
    cuda_call(&cudaEventCreateWithFlags, &start, 0);
    cuda_call(&cudaEventCreateWithFlags, &stop, 0);

    cuda_call(&cudaEventRecord, start, stream);
    f();
    cuda_call(&cudaEventRecord, stop, stream);

    cuda_call(&cudaEventSynchronize, stop);

    float milliseconds = 0;
    cuda_call(&cudaEventElapsedTime, &milliseconds, start, stop);

    return std::chrono::duration<float, std::milli>{milliseconds};
  }

  inline auto make_unique_device_pitched(const std::size_t width_in_bytes, const std::size_t height)
  {
    void* ptr;
    std::size_t pitch_in_bytes;

    using func_t = cudaError_t (*)(void**, size_t*, size_t, size_t);
    cuda_call<func_t>(&cudaMallocPitch, &ptr, &pitch_in_bytes, width_in_bytes, height);

    return std::make_tuple(std::unique_ptr<byte[], detail::cuda_free>(static_cast<byte*>(ptr)),
                           static_cast<std::int32_t>(pitch_in_bytes));
  }

  inline auto make_unique_host_pinned(const std::size_t num_bytes)
  {
    void* ptr = nullptr;

    using func_t = cudaError_t (*)(void**, size_t);
    cuda_call<func_t>(&cudaMallocHost, &ptr, num_bytes);

    return std::unique_ptr<byte[], detail::cuda_free_host>{static_cast<byte*>(ptr)};
  }

  template<typename T>
  image<T> make_host_pinned_image(const bounds<int32_t>& bounds, const int32_t alignment = 1024)
  {
    const auto row_pitch_in_bytes = int_div_ceil(std::int32_t{sizeof(T)} * bounds.width, alignment) * alignment;
    const auto num_bytes = static_cast<size_t>(row_pitch_in_bytes * bounds.height);

    return image<T>{std::shared_ptr{make_unique_host_pinned(num_bytes)}, bounds, row_pitch_in_bytes};
  }

  template<typename T>
  void* cast_to_cuda_ptr(T* ptr)
  {
    return const_cast<void*>(static_cast<const void*>(ptr));
  }
}  // namespace quxflux
