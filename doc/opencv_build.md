# OpenCV build

build instructions to build OpenCV with CUDA median filter (as used for the benchmarks in this repo)

```
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.9.0
cd ..
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 4.9.0
cd ..
```

```
openCvSource=opencv
openCVExtraModules=opencv_contrib/modules
openCvBuild=opencv/out/bin/opencv
generator=Ninja
buildType=Release

cmake $openCvSource \
 -B"$openCvBuild/" \
 -G"$generator" \
 -DCMAKE_BUILD_TYPE=$buildType \
 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
 -DWITH_NVCUVID=OFF \
 -DWITH_CUDA=ON \
 -DCUDA_FAST_MATH=ON \
 -DWITH_CUBLAS=ON \
 -DINSTALL_TESTS=OFF \
 -DINSTALL_C_EXAMPLES=OFF \
 -DBUILD_EXAMPLES=OFF \
 -DWITH_OPENGL=ON \
 -DOPENCV_EXTRA_MODULES_PATH="$openCVExtraModules/" \
 -DOPENCV_ENABLE_NONFREE=OFF \
 -DBUILD_opencv_apps=OFF \
 -DBUILD_opencv_aruco=OFF \
 -DBUILD_opencv_bgsegm=OFF \
 -DBUILD_opencv_bioinspired=OFF \
 -DBUILD_opencv_calib3d=ON \
 -DBUILD_opencv_ccalib=OFF \
 -DBUILD_opencv_core=ON \
 -DBUILD_opencv_cudaarithm=ON \
 -DBUILD_opencv_cudabgsegm=OFF \
 -DBUILD_opencv_cudacodec=OFF \
 -DBUILD_opencv_cudafeatures2d=ON \
 -DBUILD_opencv_cudafilters=ON \
 -DBUILD_opencv_cudaimgproc=ON \
 -DBUILD_opencv_cudalegacy=ON \
 -DBUILD_opencv_cudaobjdetect=OFF \
 -DBUILD_opencv_cudaoptflow=OFF \
 -DBUILD_opencv_cudastereo=OFF \
 -DBUILD_opencv_cudawarping=ON \
 -DBUILD_opencv_cudev=ON \
 -DBUILD_opencv_datasets=OFF \
 -DBUILD_opencv_dnn=OFF \
 -DBUILD_opencv_dnn_objdetect=OFF \
 -DBUILD_opencv_dnn_superres=OFF \
 -DBUILD_opencv_dpm=OFF \
 -DBUILD_opencv_face=OFF \
 -DBUILD_opencv_features2d=ON \
 -DBUILD_opencv_flann=ON \
 -DBUILD_opencv_fuzzy=ON \
 -DBUILD_opencv_gapi=ON \
 -DBUILD_opencv_hfs=ON \
 -DBUILD_opencv_highgui=ON \
 -DBUILD_opencv_img_hash=OFF \
 -DBUILD_opencv_imgcodecs=ON \
 -DBUILD_opencv_imgproc=ON \
 -DBUILD_opencv_intensity_transform=OFF \
 -DBUILD_opencv_java_bindings_generator=OFF \
 -DBUILD_opencv_js=OFF \
 -DBUILD_opencv_js_bindings_generator=OFF \
 -DBUILD_opencv_line_descriptor=OFF \
 -DBUILD_opencv_mcc=OFF \
 -DBUILD_opencv_ml=OFF \
 -DBUILD_opencv_objc_bindings_generator=OFF \
 -DBUILD_opencv_objdetect=OFF \
 -DBUILD_opencv_optflow=ON \
 -DBUILD_opencv_phase_unwrapping=OFF \
 -DBUILD_opencv_photo=ON \
 -DBUILD_opencv_plot=ON \
 -DBUILD_opencv_python_bindings_generator=OFF \
 -DBUILD_opencv_python_tests=OFF \
 -DBUILD_opencv_quality=ON \
 -DBUILD_opencv_rapid=ON \
 -DBUILD_opencv_reg=ON \
 -DBUILD_opencv_rgbd=ON \
 -DBUILD_opencv_saliency=OFF \
 -DBUILD_opencv_shape=OFF \
 -DBUILD_opencv_stereo=OFF \
 -DBUILD_opencv_stitching=ON \
 -DBUILD_opencv_structured_light=OFF \
 -DBUILD_opencv_superres=OFF \
 -DBUILD_opencv_surface_matching=ON \
 -DBUILD_opencv_text=ON \
 -DBUILD_opencv_tracking=OFF \
 -DBUILD_opencv_ts=OFF \
 -DBUILD_opencv_video=OFF \
 -DBUILD_opencv_videoio=OFF \
 -DBUILD_opencv_videostab=OFF \
 -DBUILD_opencv_world=OFF \
 -DBUILD_opencv_xfeatures2d=OFF \
 -DBUILD_opencv_ximgproc=OFF \
 -DBUILD_opencv_xobjdetect=OFF \
 -DBUILD_opencv_xphoto=OFF \
 -DCUDA_ARCH_PTX=6.0 \
 -DCUDA_GENERATION="Pascal"
```

```
cmake --build $openCvBuild --config $buildType
```