#!/usr/bin/env bash

# Usage: build.sh <os_type><os_version>[,<os_type><os_version>] <arch>[,<arch>]

REPO_PATH="konduit/jenkins-agent"
declare -A os=([""]=)
OS_TYPES=(centos ubuntu android)
OS_VERSIONS=()
PLATFORM_NAMES=()
CUDA_VERSIONS=(cuda-9.2 cuda-10.0 cuda-10.1)

for os in OS_TYPES do
    docker build -f "manifests/jenkins-agentmanifests/jenkins-agent/${os_type}/${os_version}/${arch}/Dockerfile" -t "${REPO_PATH}:${arch}-${os_type}${os_version}" .
    for cuda in CUDA_VERSIONS do
        docker_file_path="manifests/jenkins-agentmanifests/jenkins-agent/${os_type}/${os_version}/${arch}/cuda/${cuda_version}/Dockerfile"
        docker_tag="${REPO_PATH}:${arch}-${os_type}${os_version}-"
        
        docker build -f "${docker_file_path}" -t "${docker_tag}" .

docker build -f manifests/jenkins-agent/centos/6/amd64/base/Dockerfile -t konduit/jenkins-agent:amd64-centos6 .

docker build -f manifests/jenkins-agent/centos/6/amd64/cuda/9.0/Dockerfile -t konduit/jenkins-agent:amd64-centos6 .
docker build -f manifests/jenkins-agent/centos/6/amd64/cuda/9.1/Dockerfile -t konduit/jenkins-agent:amd64-centos6 .
docker build -f manifests/jenkins-agent/centos/6/amd64/cuda/9.2/Dockerfile -t konduit/jenkins-agent:amd64-centos6-cuda9.2-cudnn7 .
docker build -f manifests/jenkins-agent/centos/6/amd64/cuda/10.0/Dockerfile -t konduit/jenkins-agent:amd64-centos6-cuda10.0-cudnn7 .
docker build -f manifests/jenkins-agent/centos/6/amd64/cuda/10.1/Dockerfile -t konduit/jenkins-agent:amd64-centos6-cuda10.1-cudnn7 .
docker build -f manifests/jenkins-agent/centos/6/amd64/cuda/10.2/Dockerfile -t konduit/jenkins-agent:amd64-centos6-cuda10.2-cudnn7 .

docker build -f manifests/jenkins-agent/centos/7/amd64/base/Dockerfile -t konduit/jenkins-agent:amd64-centos7 .

docker build -f manifests/jenkins-agent/centos/7/amd64/cuda/9.2/Dockerfile -t konduit/jenkins-agent:amd64-centos7-cuda9.2-cudnn7 .
docker build -f manifests/jenkins-agent/centos/7/amd64/cuda/10.0/Dockerfile -t konduit/jenkins-agent:amd64-centos7-cuda10.0-cudnn7 .
docker build -f manifests/jenkins-agent/centos/7/amd64/cuda/10.1/Dockerfile -t konduit/jenkins-agent:amd64-centos7-cuda10.1-cudnn7 .
docker build -f manifests/jenkins-agent/centos/7/amd64/cuda/10.2/Dockerfile -t konduit/jenkins-agent:amd64-centos7-cuda10.2-cudnn7 .

docker build -f manifests/jenkins-agent/android/arm/Dockerfile -t konduit/jenkins-agent:arm-android .
docker build -f manifests/jenkins-agent/centos/7/armhf/Dockerfile -t konduit/jenkins-agent:armhf-centos7 .

docker build -f manifests/jenkins-agent/ubuntu/16.04/ppc64le/base/Dockerfile -t konduit/jenkins-agent:ppc64le-ubuntu16.04 .
docker build -f manifests/jenkins-agent/ubuntu/16.04/ppc64le/cuda/9.2/Dockerfile -t konduit/jenkins-agent:ppc64le-ubuntu16.04-cuda9.2-cudnn7 .

docker push konduit/jenkins-agent:amd64-centos6
docker push konduit/jenkins-agent:amd64-centos6-cuda9.2-cudnn7
docker push konduit/jenkins-agent:amd64-centos6-cuda10.0-cudnn7
docker push konduit/jenkins-agent:amd64-centos6-cuda10.1-cudnn7
docker push konduit/jenkins-agent:amd64-centos6-cuda10.2-cudnn7
docker push konduit/jenkins-agent:amd64-centos7
docker push konduit/jenkins-agent:amd64-centos7-cuda9.2-cudnn7
docker push konduit/jenkins-agent:amd64-centos7-cuda10.0-cudnn7
docker push konduit/jenkins-agent:amd64-centos7-cuda10.1-cudnn7
docker push konduit/jenkins-agent:amd64-centos7-cuda10.2-cudnn7
docker push konduit/jenkins-agent:arm-android
docker push konduit/jenkins-agent:armhf-centos7

docker pull docker.ci.konduit.ai/konduit/jenkins-agent:amd64-centos6
docker pull docker.ci.konduit.ai/konduit/jenkins-agent:amd64-centos6-cuda9.2-cudnn7
docker pull docker.ci.konduit.ai/konduit/jenkins-agent:amd64-centos6-cuda10.0-cudnn7
docker pull docker.ci.konduit.ai/konduit/jenkins-agent:amd64-centos6-cuda10.1-cudnn7
docker pull docker.ci.konduit.ai/konduit/jenkins-agent:amd64-centos6-cuda10.2-cudnn7
docker pull docker.ci.konduit.ai/konduit/jenkins-agent:amd64-centos7
docker pull docker.ci.konduit.ai/konduit/jenkins-agent:amd64-centos7-cuda9.2-cudnn7
docker pull docker.ci.konduit.ai/konduit/jenkins-agent:amd64-centos7-cuda10.0-cudnn7
docker pull docker.ci.konduit.ai/konduit/jenkins-agent:amd64-centos7-cuda10.1-cudnn7
docker pull docker.ci.konduit.ai/konduit/jenkins-agent:amd64-centos7-cuda10.2-cudnn7
docker pull docker.ci.konduit.ai/konduit/jenkins-agent:arm-android
docker pull docker.ci.konduit.ai/konduit/jenkins-agent:armhf-centos7

docker image rm konduit/jenkins-agent:amd64-centos6
docker image rm konduit/jenkins-agent:amd64-centos6-cuda9.2-cudnn7
docker image rm konduit/jenkins-agent:amd64-centos6-cuda10.0-cudnn7
docker image rm konduit/jenkins-agent:amd64-centos6-cuda10.1-cudnn7
docker image rm konduit/jenkins-agent:amd64-centos6-cuda10.2-cudnn7
docker image rm konduit/jenkins-agent:amd64-centos7
docker image rm konduit/jenkins-agent:amd64-centos7-cuda9.2-cudnn7
docker image rm konduit/jenkins-agent:amd64-centos7-cuda10.0-cudnn7
docker image rm konduit/jenkins-agent:amd64-centos7-cuda10.1-cudnn7
docker image rm konduit/jenkins-agent:amd64-centos7-cuda10.2-cudnn7
docker image rm konduit/jenkins-agent:arm-android
docker image rm konduit/jenkins-agent:armhf-centos7
