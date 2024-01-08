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

#include <nppi_filtering_functions.h>

#include <abstract_filter_impl.h>
#include <filter_database.h>
#include <filter_spec_factory.h>

#include <shared/cuda/gpu_image.h>
#include <shared/cuda/image_transfer.h>
#include <shared/cuda/stream_handle.h>

#include <npp.h>

#include <stdexcept>
#include <utility>

namespace quxflux
{
  namespace
  {
    template<typename F, typename... Args>
    void npp_call(const F& f, Args&&... args)
    {
      const NppStatus r = f(std::forward<Args>(args)...);

      if (r != NPP_NO_ERROR)
        throw std::runtime_error("Call to npp failed");
    }

    template<typename FilterSpec>
    struct filter_impl : abstract_filter_impl
    {
      using T = typename FilterSpec::value_type;

      static ::quxflux::filter_configuration filter_configuration()
      {
        return {"npp", typeid(typename FilterSpec::value_type), FilterSpec::filter_size, ""};
      };

      void filter(const std::any& source_image, std::any& target_image) override
      {
        const auto& source = std::any_cast<const image<T>&>(source_image);
        auto& target = std::any_cast<image<T>&>(target_image);

        const auto bounds = source.bounds();

        if (gpu_img_source.bounds() != bounds)
        {
          gpu_img_source = make_gpu_image<T>(bounds);
          gpu_img_target = make_gpu_image<T>(bounds);
        }

        stream_handle stream;

        NppStreamContext npp_stream_ctxt{};
        npp_call(&nppGetStreamContext, &npp_stream_ctxt);
        npp_stream_ctxt.hStream = stream;
        npp_stream_ctxt.nStreamFlags = stream.get_flags();

        // the benchmark actually runs on a smaller portion of the input image because npp does not offer
        // any special treatment for the boundaries

        constexpr std::int32_t filter_radius = FilterSpec::filter_size / 2;

        const NppiSize roi{bounds.width - FilterSpec::filter_size, bounds.height - FilterSpec::filter_size};
        constexpr NppiSize filter_size{FilterSpec::filter_size, FilterSpec::filter_size};
        constexpr NppiPoint anchor{filter_radius, filter_radius};

        {
          Npp32u buffer_size = 0;
          npp_call(&nppiFilterMedianGetBufferSize_8u_C1R_Ctx, roi, filter_size, &buffer_size, npp_stream_ctxt);

          if (buffer_size > scratch_buffer_size)
          {
            scratch_buffer_size = buffer_size;
            scratch_buffer = make_unique_host_pinned(buffer_size);
          }
        }

        transfer(source, gpu_img_source, stream);

        const auto src_pitch = gpu_img_source.row_pitch_in_bytes();
        const auto dst_pitch = gpu_img_target.row_pitch_in_bytes();

        npp_call(&nppiFilterMedian_8u_C1R_Ctx,
                 gpu_img_source.data_ptr() + src_pitch * filter_radius + int{sizeof(T)} * filter_radius, src_pitch,
                 gpu_img_target.data_ptr() + dst_pitch * filter_radius + int{sizeof(T)} * filter_radius, dst_pitch, roi,
                 filter_size, anchor, scratch_buffer.get(), npp_stream_ctxt);

        transfer(gpu_img_target, target, stream);

        stream.synchronize();
      }

      std::size_t scratch_buffer_size{};
      unique_pinned_host_ptr scratch_buffer;

      gpu_image<T> gpu_img_source;
      gpu_image<T> gpu_img_target;
    };

    auto filter_registrar = [] {
      for_each_filter_spec([&](auto filter_spec) {
        using filter_spec_t = decltype(filter_spec);

        using impl_t = filter_impl<filter_spec_t>;
        register_filter({impl_t::filter_configuration(), std::make_shared<impl_t>()});
      });

      return 0;
    }();
  }  // namespace
}  // namespace quxflux
