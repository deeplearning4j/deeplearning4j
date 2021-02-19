# Build instructions for Jetson Docker container

## Build system config

### General setup steps

Old version

```
sudo apt-get install qemu binfmt-support qemu-user-static
sudo cat /proc/sys/fs/binfmt_misc/status
enabled
```

New version

```
sudo apt-get install build-essential pkg-config git libglib2.0-dev libfdt-dev libpixman-1-dev zlib1g-dev checkinstall binfmt-support

sudo cat /proc/sys/fs/binfmt_misc/status
Output: enabled

wget https://download.qemu.org/qemu-4.2.0.tar.xz
tar xvJf qemu-4.2.0.tar.xz
cd qemu-4.2.0
./configure
make

sudo checkinstall make install
sudo dpkg -i ./qemu_4.2.0-1_amd64.deb

qemu-system-x86_64 --version

git clone git://git.qemu.org/qemu.git
cd qemu
git submodule update --init --recursive
git checkout v4.2.0
./configure \
    --prefix=$(cd ..; pwd)/qemu-user-static \
    --static \
    --disable-system \
    --enable-linux-user
make -j2
make install
cd ../qemu-user-static/bin
for i in *; do cp $i $i-static; done
```

Links:
* https://www.qemu.org/download/#source
* https://wiki.qemu.org/Hosts/Linux
* https://askubuntu.com/questions/1067722/how-do-i-install-qemu-3-0-on-ubuntu-18-04
* https://mathiashueber.com/manually-update-qemu-on-ubuntu-18-04/
* https://github.com/multiarch/qemu-user-static/issues/18
* http://logan.tw/posts/2018/02/18/build-qemu-user-static-from-source-code/
* https://github.com/multiarch/qemu-user-static/issues/18


### Enable qemu multiarch

[Enable qemu multiarch support](https://github.com/multiarch/qemu-user-static#getting-started)...

```
vim /etc/systemd/system/docker-qemu-binfmt-fix.service

[Unit]
Description=Qemu static binfmt_misc fix
Requires=docker.service
After=docker.service

[Service]
Restart=no
ExecStart=/usr/bin/docker run --name qemu-static-binfmt-fix --rm --privileged multiarch/qemu-user-static --reset -p yes

[Install]
WantedBy=default.target

sudo systemctl enable docker-qemu-binfmt-fix.service

docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

cat /proc/sys/fs/binfmt_misc/qemu-aarch64
...
flags: OCF | flags: F
...
```

### NVIDIA SDK Manager

[Download](https://developer.nvidia.com/nvsdk-manager) system specific package for NVIDIA SDK Manager and install it or use NVIDIA SDK Manager in [Docker](https://github.com/atinfinity/sdk_manager_docker).

[WARNING] NVIDIA SDK Manager requires GUI to work properly.


## Base build steps

```
git clone https://github.com/idavis/jetson-containers.git
cd jetson-containers
make deps-32.2-nano-jetpack-4.2.1
make l4t-32.2.1-nano
make 32.2.1-nano-jetpack-4.2.2
```

```
git clone https://github.com/KonduitAI/docker.git
cd manifests/jenkins-agent/ubuntu/18.04/arm64/cuda/10.0/Dockerfile
docker build -t konduit/jenkins-agent:arm64-nano-ubuntu18.04-l4t32.2.1-jetpack4.2.2-cuda10.0-cudnn7 .
```

## Optional/fix build steps

```
vim Makefile
# DOCKER_BUILD_ARGS ?= "" -> DOCKER_BUILD_ARGS ?= ""
```

```
vim docker/jetpack/4.2.2/nano/devel/Dockerfile
RUN apt-get update && apt-get install -y \ -> RUN apt-get update && apt-get install -y --no-install-recommends \
```

```
vim docker/l4t/32.2.1/nano/Dockerfile
# echo "2d648bbc77c510c4e7e0c809996d24e8 *./${DRIVER_PACK}" | md5sum -c - && \ -> echo "2d648bbc77c510c4e7e0c809996d24e8 *./${DRIVER_PACK}" | md5sum -c - && \
```

## Verify Docker image

```
git clone https://github.com/KonduitAI/deeplearning4j.git
cd deeplearning4j/libnd4j
./buildnativeoperations.sh --build-type release --chip cpu --platform linux-arm64 --chip-extension '' --chip-version '' --compute '' '' -j 8
```

## Jetson containers info

### Main articles
1. https://codepyre.com/2019/12/arming-yourself/
2. https://github.com/NVIDIA/nvidia-docker/wiki/NVIDIA-Container-Runtime-on-Jetson
3. https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano-in-nvidia-docker-container/
4. https://github.com/idavis/jetson-containers
5. https://dev.to/azure/supercharge-your-containerized-iot-workloads-with-gpu-acceleration-on-nvidia-jetson-devices-4532
6. https://dev.to/azure/building-jetson-containers-for-nvidia-devices-on-windows-10-with-vs-code-and-wsl-v2-1ao
7. https://dev.to/monkeycoder99/comment/j74g

### Additional articles
5. https://ownyourbits.com/2018/06/27/running-and-building-arm-docker-containers-in-x86/
6. https://www.balena.io/blog/building-arm-containers-on-any-x86-machine-even-dockerhub/
7. https://www.96boards.org/documentation/guides/crosscompile/commandline.html
8. http://jensd.be/800/linux/cross-compiling-for-arm-with-ubuntu-16-04-lts
9. https://medium.com/@w.wilson.antoine/how-to-create-cuda-enabled-container-for-nvidia-jetson-adf493f8df3e
10. https://ngc.nvidia.com/catalog/containers/nvidia:l4t-base

### Cross platform compilation (outdated)
1. https://docs.nvidia.com/jetson/l4t-multimedia/cross_platform_support.html
2. https://devtalk.nvidia.com/default/topic/1051466/jetson-nano/docker-image-for-cross-compilation-/1