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

#include <abstract_filter_impl.h>

#include <impls/opencv/cv_mat_image_view.h>

#include <filter_spec_factory.h>
#include <filter_database.h>

#include <shared/image.h>
#include <shared/cuda/util.h>

#include <stdexcept>

namespace quxflux
{
  namespace
  {
    std::pair<int, int> get_device_compute_capability()
    {
      int device_id = cv::cuda::getDevice();
      if (device_id < 0)
        throw std::runtime_error("cv::cuda::getDevice failed");

      int major = 0, minor = 0;
      cuda_call(&cudaDeviceGetAttribute, &major, cudaDevAttrComputeCapabilityMajor, device_id);
      cuda_call(&cudaDeviceGetAttribute, &minor, cudaDevAttrComputeCapabilityMinor, device_id);

      return {major, minor};
    }

    bool isDeviceCompatible()
    {
      const auto [major, minor] = get_device_compute_capability();

      if (cv::cuda::TargetArchs::hasEqualOrLessPtx(major, minor))
        return true;

      for (int i = minor; i >= 0; i--)
        if (cv::cuda::TargetArchs::hasBin(major, i))
          return true;

      return false;
    }

    template<typename FilterSpec>
    struct filter_impl : abstract_filter_impl
    {
      static ::quxflux::filter_configuration filter_configuration()
      {
        return {"opencv_cuda", typeid(typename FilterSpec::value_type), FilterSpec::filter_size, ""};
      };

      void filter(const std::any& source_image, std::any& target_image) override
      {
        using T = typename FilterSpec::value_type;

        const auto& source = std::any_cast<const image<T>&>(source_image);
        auto& target = std::any_cast<image<T>&>(target_image);

        const auto cv_source = create_mat_view_for_image(source);
        const auto cv_target = create_mat_view_for_image(target);

        if (gpu_mat_source.size() != cv_source.size())
        {
          gpu_mat_source = cv::cuda::GpuMat(cv_source.size(), cv_source.type());
          gpu_mat_target = cv::cuda::GpuMat(cv_target.size(), cv_target.type());
        }

        if (!cv_filter)
        {
          cv_filter = cv::cuda::createMedianFilter(cv_source.type(), FilterSpec::filter_size);
        }

        cv::cuda::Stream stream(cudaStreamNonBlocking);

        gpu_mat_source.upload(cv_source, stream);
        cv_filter->apply(gpu_mat_source, gpu_mat_target, stream);
        gpu_mat_target.download(cv_target, stream);

        stream.waitForCompletion();
      }

      cv::Ptr<cv::cuda::Filter> cv_filter;
      cv::cuda::GpuMat gpu_mat_source;
      cv::cuda::GpuMat gpu_mat_target;
    };

    [[maybe_unused]] auto filter_registrar = [] {
      if (!isDeviceCompatible())
        throw std::runtime_error{"Device is unsupported for used OpenCV version"};

      for_each_filter_spec([&](auto filter_spec) {
        using filter_spec_t = decltype(filter_spec);

        if constexpr (cv_mat_supports_type<typename filter_spec_t::value_type>::value)
        {
          if constexpr (cv::DataType<typename filter_spec_t::value_type>::type ==
                        CV_8UC1)  // according to the documentation only CV_8UC1 is supported as of OpenCV 4.5.5
          {
            using impl_t = filter_impl<filter_spec_t>;
            register_filter({impl_t::filter_configuration(), std::make_shared<filter_impl<filter_spec_t>>()});
          }
        }
      });

      return 0;
    }();
  }  // namespace
}  // namespace quxflux
