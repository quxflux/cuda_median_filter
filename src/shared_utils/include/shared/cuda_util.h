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
#include <memory>
#include <stdexcept>
#include <tuple>

#include <cuda_median_filter/detail/cuda_wrap.h>
#include <cuda_median_filter/detail/primitives.h>

namespace quxflux
{
  namespace detail
  {
    struct cuda_free
    {
      template<typename T>
      constexpr void operator()(T* device_ptr) const
      {
        [[maybe_unused]] const cudaError_t r = cudaFree(device_ptr);

        assert(r == cudaSuccess && "cudaFree failed!");
      }
    };

    struct cuda_free_host
    {
      template<typename T>
      constexpr void operator()(T* ptr) const
      {
        [[maybe_unused]] const cudaError_t r = cudaFreeHost(ptr);

        assert(r == cudaSuccess && "cudaFree failed!");
      }
    };
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

  using unique_pitched_device_ptr = std::unique_ptr<std::byte, detail::cuda_free>;

  inline auto make_unique_device_pitched(const std::size_t width_in_bytes, const std::size_t height)
  {
    void* ptr;
    std::size_t pitch_in_bytes;

    using func_t = cudaError_t (*)(void**, size_t*, size_t, size_t);
    cuda_call<func_t>(&cudaMallocPitch, &ptr, &pitch_in_bytes, width_in_bytes, height);

    return std::make_tuple(std::unique_ptr<std::byte, detail::cuda_free>(static_cast<std::byte*>(ptr)),
                           static_cast<std::int32_t>(pitch_in_bytes));
  }

  using unique_pinned_host_ptr = std::unique_ptr<std::byte, detail::cuda_free_host>;

  inline unique_pinned_host_ptr make_unique_host_pinned(const std::size_t num_bytes)
  {
    void* ptr = nullptr;

    using func_t = cudaError_t (*)(void**, size_t);
    cuda_call<func_t>(&cudaMallocHost, &ptr, num_bytes);

    return unique_pinned_host_ptr(static_cast<std::byte*>(ptr));
  }

  template<typename T>
  void* cast_to_cuda_ptr(T* ptr)
  {
    return const_cast<void*>(static_cast<const void*>(ptr));
  }
}  // namespace quxflux
