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
// CPU stub implementations of DeviceManager member functions.
// This file is only compiled for CPU builds (execution/cpu/ is excluded from CUDA builds).
//

#include <execution/DeviceManager.h>

namespace sd {
namespace modelparallel {

void DeviceManager::discoverCudaDevices() {
    // No CUDA devices to discover in CPU build
}

void DeviceManager::initializeP2PConnections() {
    // No P2P connections in CPU build
}

void DeviceManager::probeP2PCapabilities(int device1, int device2) {
    // No P2P probing in CPU build
}

bool DeviceManager::enableP2P(int device1, int device2) {
    return false;
}

void DeviceManager::disableP2P(int device1, int device2) {
    // No P2P in CPU build
}

void DeviceManager::updateMemoryStats(int globalIndex) {
    // CPU build: no GPU memory stats to update
}

void DeviceManager::setCurrentDeviceCuda(const DeviceInfo& device) {
    // No CUDA device switching in CPU build
}

void DeviceManager::synchronizeAll() {
    // No GPU synchronization in CPU build
}

void DeviceManager::synchronize(int globalIndex) {
    // No GPU synchronization in CPU build
}

}  // namespace modelparallel
}  // namespace sd
