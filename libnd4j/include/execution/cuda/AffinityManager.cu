/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//

#include <logger.h>
#include <execution/AffinityManager.h>
#include <exceptions/cuda_exception.h>
#include <LaunchContext.h>

thread_local int globalThreadToDevice = -1;

namespace nd4j {
    std::mutex AffinityManager::_currentMutex;
    std::mutex AffinityManager::_numberMutex;
    int AffinityManager::_numberOfDevices = -1;

    int AffinityManager::currentDeviceId() {
        // if there's no affinity set - set it now
        if (globalThreadToDevice < 0) {

            // this block must be thread-local
            _currentMutex.lock();

            globalThreadToDevice = _lastDevice++;

            // we need to check if we've got deviceId >= number of actual devices, and reset to zero otherwise
            if (globalThreadToDevice >= numberOfDevices()) {
                globalThreadToDevice = 0;
                _lastDevice = numberOfDevices() > 1 ? 1 : 0;
            }

            _currentMutex.unlock();

            setCurrentNativeDevice(globalThreadToDevice);
        }

        // if we already know affinity - just return it
        if (globalThreadToDevice >= 0)
            return globalThreadToDevice;

        int dev = 0;
        auto res = cudaGetDevice(&dev);

        if (res != 0)
            throw cuda_exception::build("cudaGetDevice failed", res);

        return dev;
    }

    int AffinityManager::currentNativeDeviceId() {
        int dev = 0;
        auto res = cudaGetDevice(&dev);

        if (res != 0)
            throw cuda_exception::build("cudaGetDevice failed", res);

        return dev;
    }

    int AffinityManager::numberOfDevices() {
        _numberMutex.lock();
        // we want to cache number of devices
        if (_numberOfDevices <= 0) {
            int dev = 0;
            auto res = cudaGetDeviceCount(&dev);

            if (res != 0)
                throw cuda_exception::build("cudaGetDeviceCount failed", res);

            _numberOfDevices = dev;
        }
        _numberMutex.unlock();

        return _numberOfDevices;
    }

    void AffinityManager::setCurrentNativeDevice(int deviceId) {
        auto res = cudaSetDevice(deviceId);
        if (res != 0)
            throw cuda_exception::build("setCurrentDevice failed", res);
    }

    void AffinityManager::setCurrentDevice(int deviceId) {
        auto previousDeviceId = globalThreadToDevice;
        if (previousDeviceId >= 0 && LaunchContext::isInitialized()) {
            auto res = cudaStreamSynchronize(*LaunchContext::defaultContext()->getCudaStream());
            if (res != 0)
                throw cuda_exception::build("setCurrentDevice -> sync failed", res);

            res = cudaStreamSynchronize(*LaunchContext::defaultContext()->getCudaSpecialStream());
            if (res != 0)
                throw cuda_exception::build("setCurrentDevice -> specialSync failed", res);

            if (deviceId != previousDeviceId) {
                // discard existing stuff
                nd4j_printf("AffinityManager::setCurrentDevice() was invoked, releasing buffers\n", "");
                LaunchContext::releaseBuffers();
            }
        }

        if (deviceId != previousDeviceId) {
            auto res = cudaSetDevice(deviceId);
            if (res != 0)
                throw cuda_exception::build("cudaSetDevice failed", res);
        }

        // update thread-device affinity
        globalThreadToDevice = deviceId;
    }

    std::atomic<int> AffinityManager::_lastDevice;// = std::atomic<int>(initialV);
}