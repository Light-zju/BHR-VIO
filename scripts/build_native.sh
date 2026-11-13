#!/usr/bin/env bash
set -euo pipefail
# CPU portability only; no estimation factors, thresholds or source operators change.
find /opt/ORB_SLAM3 -name CMakeLists.txt -exec sed -i 's/-march=native/-march=haswell -mtune=generic/g' {} +
sed '/#Old examples/,$d' /opt/ORB_SLAM3/CMakeLists.txt > /opt/ORB_SLAM3/CMakeLists.release
mv /opt/ORB_SLAM3/CMakeLists.release /opt/ORB_SLAM3/CMakeLists.txt
cmake -S /opt/Pangolin -B /opt/Pangolin/build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_TESTS=OFF -DBUILD_PANGOLIN_PYTHON=OFF
cmake --build /opt/Pangolin/build -j2
cmake --install /opt/Pangolin/build
ldconfig
for component in DBoW2 g2o; do
 cmake -S /opt/ORB_SLAM3/Thirdparty/$component -B /opt/ORB_SLAM3/Thirdparty/$component/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=11 -DCMAKE_CXX_FLAGS=-std=c++11
 cmake --build /opt/ORB_SLAM3/Thirdparty/$component/build -j2
done
cmake -S /opt/ORB_SLAM3 -B /opt/ORB_SLAM3/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=11 -DCMAKE_CXX_FLAGS=-std=c++11
cmake --build /opt/ORB_SLAM3/build --target stereo_inertial_euroc -j2
tar -xzf /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz -C /opt/ORB_SLAM3/Vocabulary
