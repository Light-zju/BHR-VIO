FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake pkg-config python3 libopencv-dev libeigen3-dev libboost-serialization-dev libssl-dev libglew-dev libgl1-mesa-dev libegl1-mesa-dev libepoxy-dev libjpeg-dev libpng-dev ca-certificates && rm -rf /var/lib/apt/lists/*
COPY native/Pangolin /opt/Pangolin
COPY native/ORB_SLAM3 /opt/ORB_SLAM3
COPY scripts/build_native.sh /opt/build_native.sh
RUN bash /opt/build_native.sh
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/ORB_SLAM3/lib:/opt/ORB_SLAM3/Thirdparty/DBoW2/lib:/opt/ORB_SLAM3/Thirdparty/g2o/lib
WORKDIR /output
