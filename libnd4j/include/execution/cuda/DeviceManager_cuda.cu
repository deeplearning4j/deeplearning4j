/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// CUDA implementations of DeviceManager member functions.
// Extracted from execution/impl/DeviceManager.cpp
//

#ifdef SD_CUDA

#include <execution/DeviceManager.h>
#include <cuda_runtime.h>

namespace sd {
namespace modelparallel {

void DeviceManager::discoverCudaDevices() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        return;
    }

    // Save current device to restore after enumeration
    int savedDevice = -1;
    cudaGetDevice(&savedDevice);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, i);
        if (err != cudaSuccess) {
            continue;
        }

        DeviceInfo gpuInfo;
        gpuInfo.type = DeviceType::CUDA_GPU;
        gpuInfo.deviceIndex = i;
        gpuInfo.globalIndex = static_cast<int>(_devices.size());
        gpuInfo.name = props.name;

        gpuInfo.totalMemory = props.totalGlobalMem;

        // Get free memory
        size_t freeMem, totalMem;
        cudaSetDevice(i);
        cudaMemGetInfo(&freeMem, &totalMem);
        gpuInfo.freeMemory = freeMem;
        gpuInfo.availableMemory = freeMem;

        gpuInfo.computeCapabilityMajor = props.major;
        gpuInfo.computeCapabilityMinor = props.minor;
        gpuInfo.numSMs = props.multiProcessorCount;
        gpuInfo.clockSpeedGHz = props.clockRate / 1e6f;

        // Check unified memory support (Pascal and later)
        gpuInfo.supportsUnifiedMemory = (props.major >= 6);
        gpuInfo.supportsAsyncCopy = true;
        gpuInfo.available = true;
        gpuInfo.engine = samediff::ENGINE_CUDA;

        _devices.push_back(gpuInfo);
        _devicesByType[DeviceType::CUDA_GPU].push_back(gpuInfo.globalIndex);
    }

    // Restore the device that was active before enumeration
    cudaSetDevice(savedDevice);
}

void DeviceManager::initializeP2PConnections() {
    auto& cudaDevices = _devicesByType[DeviceType::CUDA_GPU];

    for (size_t i = 0; i < cudaDevices.size(); ++i) {
        for (size_t j = i + 1; j < cudaDevices.size(); ++j) {
            int dev1 = _devices[cudaDevices[i]].deviceIndex;
            int dev2 = _devices[cudaDevices[j]].deviceIndex;

            probeP2PCapabilities(dev1, dev2);
        }
    }
}

void DeviceManager::probeP2PCapabilities(int device1, int device2) {
    int canAccessPeer12 = 0, canAccessPeer21 = 0;

    cudaDeviceCanAccessPeer(&canAccessPeer12, device1, device2);
    cudaDeviceCanAccessPeer(&canAccessPeer21, device2, device1);

    if (canAccessPeer12 || canAccessPeer21) {
        P2PConnection conn;
        conn.sourceDevice = device1;
        conn.targetDevice = device2;
        conn.directAccess = (canAccessPeer12 && canAccessPeer21);
        conn.bidirectional = (canAccessPeer12 && canAccessPeer21);

        // Check NVLink (if supported)
        // This is a simplified check - actual NVLink detection is more complex
        conn.nvlinkConnected = false;

        _p2pConnections.push_back(conn);

        // Update device info
        int globalIdx1 = getGlobalIndex(samediff::ENGINE_CUDA, device1);
        int globalIdx2 = getGlobalIndex(samediff::ENGINE_CUDA, device2);

        if (globalIdx1 >= 0) {
            _devices[globalIdx1].supportsP2P = true;
            _devices[globalIdx1].p2pConnectedDevices.push_back(globalIdx2);
        }
        if (globalIdx2 >= 0) {
            _devices[globalIdx2].supportsP2P = true;
            _devices[globalIdx2].p2pConnectedDevices.push_back(globalIdx1);
        }
    }
}

bool DeviceManager::enableP2P(int device1, int device2) {
    cudaError_t err1 = cudaSetDevice(device1);
    if (err1 != cudaSuccess) return false;

    cudaError_t err2 = cudaDeviceEnablePeerAccess(device2, 0);
    if (err2 != cudaSuccess && err2 != cudaErrorPeerAccessAlreadyEnabled) {
        return false;
    }

    err1 = cudaSetDevice(device2);
    if (err1 != cudaSuccess) return false;

    err2 = cudaDeviceEnablePeerAccess(device1, 0);
    if (err2 != cudaSuccess && err2 != cudaErrorPeerAccessAlreadyEnabled) {
        return false;
    }

    return true;
}

void DeviceManager::disableP2P(int device1, int device2) {
    cudaSetDevice(device1);
    cudaDeviceDisablePeerAccess(device2);

    cudaSetDevice(device2);
    cudaDeviceDisablePeerAccess(device1);
}

void DeviceManager::updateMemoryStats(int globalIndex) {
    if (auto* device = findDevice(globalIndex)) {
        if (device->type == DeviceType::CUDA_GPU) {
            size_t freeMem, totalMem;
            cudaSetDevice(device->deviceIndex);
            cudaMemGetInfo(&freeMem, &totalMem);
            device->freeMemory = freeMem;
            device->totalMemory = totalMem;

            auto it = _reservedMemory.find(globalIndex);
            size_t reserved = (it != _reservedMemory.end()) ? it->second : 0;
            device->availableMemory = (freeMem > reserved) ? (freeMem - reserved) : 0;
        }
    }
}

void DeviceManager::setCurrentDeviceCuda(const DeviceInfo& device) {
    if (device.type == DeviceType::CUDA_GPU) {
        cudaSetDevice(device.deviceIndex);
    }
}

void DeviceManager::synchronizeAll() {
    for (const auto& device : _devices) {
        if (device.type == DeviceType::CUDA_GPU) {
            cudaSetDevice(device.deviceIndex);
            cudaDeviceSynchronize();
        }
    }
}

void DeviceManager::synchronize(int globalIndex) {
    if (const auto* device = findDevice(globalIndex)) {
        if (device->type == DeviceType::CUDA_GPU) {
            cudaSetDevice(device->deviceIndex);
            cudaDeviceSynchronize();
        }
    }
}

}  // namespace modelparallel
}  // namespace sd

#endif  // SD_CUDA
