/* ******************************************************************************
 *
 * Copyright (c) 2024 Konduit K.K.
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

#ifndef SD_GENERIC_RESOURCE_MANAGER_H_
#define SD_GENERIC_RESOURCE_MANAGER_H_

#include <system/common.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>

namespace sd {
namespace generic {

class ResourceManager {
private:
    mutable std::atomic<size_t> _totalNodes{0};
    std::atomic<size_t> _activeOperations{0};
    std::chrono::steady_clock::time_point _lastCleanup;
    std::mutex _cleanupMutex;
    static constexpr size_t CLEANUP_THRESHOLD = 1000;
    static constexpr auto CLEANUP_INTERVAL = std::chrono::minutes(5);

public:
    class OperationScope {
    private:
        ResourceManager& _manager;
    public:
        explicit OperationScope(ResourceManager& m) : _manager(m) {
            _manager._activeOperations.fetch_add(1, std::memory_order_acquire);
        }
        ~OperationScope() {
            _manager._activeOperations.fetch_sub(1, std::memory_order_release);
        }
    };

    void registerNode() const {
        _totalNodes.fetch_add(1, std::memory_order_relaxed);
    }

    void unregisterNode() const {
        _totalNodes.fetch_sub(1, std::memory_order_relaxed);
    }

    OperationScope createScope() const {
        ResourceManager &m = const_cast<ResourceManager&>(*this);
        return OperationScope(m);
    }

    size_t activeOperations() const {
        return _activeOperations.load(std::memory_order_relaxed);
    }

    size_t totalNodes() const {
        return _totalNodes.load(std::memory_order_relaxed);
    }

    void resetCleanupTimer() {
        std::lock_guard<std::mutex> lock(_cleanupMutex);
        _lastCleanup = std::chrono::steady_clock::now();
    }

    bool shouldCleanup() {
        if (_totalNodes.load(std::memory_order_relaxed) < CLEANUP_THRESHOLD)
            return false;

        auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(_cleanupMutex);
        if (now - _lastCleanup >= CLEANUP_INTERVAL) {
            _lastCleanup = now;
            return true;
        }
        return false;
    }
};

}} // namespace sd::generic

#endif // SD_GENERIC_RESOURCE_MANAGER_H_