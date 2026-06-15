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

#include <helpers/TransferMetrics.h>
#include <helpers/logger.h>
#include <sstream>
#include <iomanip>

namespace sd {

// Static member initialization
TransferMetrics* TransferMetrics::_instance = nullptr;
std::mutex TransferMetrics::_instanceMutex;

TransferMetrics& TransferMetrics::getInstance() {
    if (_instance == nullptr) {
        std::lock_guard<std::mutex> lock(_instanceMutex);
        if (_instance == nullptr) {
            _instance = new TransferMetrics();
        }
    }
    return *_instance;
}

void TransferMetrics::recordTransfer(TransferType type, uint64_t bytes, uint64_t timeNs,
                                     int srcDevice, int dstDevice) {
    if (!_enabled.load()) return;

    // Update type-specific stats
    switch (type) {
        case TransferType::HOST_TO_DEVICE:
            _h2dCount++;
            _h2dBytes += bytes;
            _h2dTimeNs += timeNs;
            break;
        case TransferType::DEVICE_TO_HOST:
            _d2hCount++;
            _d2hBytes += bytes;
            _d2hTimeNs += timeNs;
            break;
        case TransferType::DEVICE_TO_DEVICE:
            _d2dCount++;
            _d2dBytes += bytes;
            _d2dTimeNs += timeNs;
            break;
        case TransferType::PEER_TO_PEER:
            _p2pCount++;
            _p2pBytes += bytes;
            _p2pTimeNs += timeNs;
            break;
    }

    // Update route-specific stats if devices specified
    if (srcDevice >= -1 && dstDevice >= -1) {
        std::string routeKey = makeRouteKey(srcDevice, dstDevice);
        std::lock_guard<std::mutex> lock(_routeMutex);
        auto& stats = _routeStats[routeKey];
        stats.transferCount++;
        stats.totalBytes += bytes;
        stats.totalTimeNs += timeNs;
    }

    // Log if enabled and above threshold
    if (_logTransfers.load() && bytes >= _minBytesForLogging.load()) {
        double mbytes = bytes / (1024.0 * 1024.0);
        double ms = timeNs / 1000000.0;
        double gbps = (bytes > 0 && timeNs > 0) ? (bytes / 1e9) / (timeNs / 1e9) : 0.0;

        const char* typeStr = "";
        switch (type) {
            case TransferType::HOST_TO_DEVICE: typeStr = "H2D"; break;
            case TransferType::DEVICE_TO_HOST: typeStr = "D2H"; break;
            case TransferType::DEVICE_TO_DEVICE: typeStr = "D2D"; break;
            case TransferType::PEER_TO_PEER: typeStr = "P2P"; break;
        }

        sd_printf("Transfer [%s]: %.2f MB in %.2f ms (%.2f GB/s) src=%d dst=%d\n",
                  typeStr, mbytes, ms, gbps, srcDevice, dstDevice);
    }
}

void TransferMetrics::recordOpExecution(uint64_t timeNs) {
    if (_enabled.load()) {
        _totalOpTimeNs += timeNs;
    }
}

uint64_t TransferMetrics::getTotalBytes(TransferType type) const {
    switch (type) {
        case TransferType::HOST_TO_DEVICE: return _h2dBytes.load();
        case TransferType::DEVICE_TO_HOST: return _d2hBytes.load();
        case TransferType::DEVICE_TO_DEVICE: return _d2dBytes.load();
        case TransferType::PEER_TO_PEER: return _p2pBytes.load();
        default: return 0;
    }
}

uint64_t TransferMetrics::getTransferCount(TransferType type) const {
    switch (type) {
        case TransferType::HOST_TO_DEVICE: return _h2dCount.load();
        case TransferType::DEVICE_TO_HOST: return _d2hCount.load();
        case TransferType::DEVICE_TO_DEVICE: return _d2dCount.load();
        case TransferType::PEER_TO_PEER: return _p2pCount.load();
        default: return 0;
    }
}

uint64_t TransferMetrics::getTotalTimeNs(TransferType type) const {
    switch (type) {
        case TransferType::HOST_TO_DEVICE: return _h2dTimeNs.load();
        case TransferType::DEVICE_TO_HOST: return _d2hTimeNs.load();
        case TransferType::DEVICE_TO_DEVICE: return _d2dTimeNs.load();
        case TransferType::PEER_TO_PEER: return _p2pTimeNs.load();
        default: return 0;
    }
}

double TransferMetrics::getAverageBandwidthGBps(TransferType type) const {
    uint64_t bytes = getTotalBytes(type);
    uint64_t timeNs = getTotalTimeNs(type);
    if (bytes == 0 || timeNs == 0) return 0.0;
    return (bytes / 1e9) / (timeNs / 1e9);
}

double TransferMetrics::getOverheadPercent() const {
    uint64_t totalOpTime = _totalOpTimeNs.load();
    if (totalOpTime == 0) return 0.0;

    uint64_t totalTransferTime = getTotalTimeNsAllTypes();
    return (totalTransferTime * 100.0) / totalOpTime;
}

uint64_t TransferMetrics::getTotalBytesAllTypes() const {
    return _h2dBytes.load() + _d2hBytes.load() + _d2dBytes.load() + _p2pBytes.load();
}

uint64_t TransferMetrics::getTotalTimeNsAllTypes() const {
    return _h2dTimeNs.load() + _d2hTimeNs.load() + _d2dTimeNs.load() + _p2pTimeNs.load();
}

void TransferMetrics::reset() {
    _h2dCount = 0;
    _h2dBytes = 0;
    _h2dTimeNs = 0;

    _d2hCount = 0;
    _d2hBytes = 0;
    _d2hTimeNs = 0;

    _d2dCount = 0;
    _d2dBytes = 0;
    _d2dTimeNs = 0;

    _p2pCount = 0;
    _p2pBytes = 0;
    _p2pTimeNs = 0;

    _totalOpTimeNs = 0;

    std::lock_guard<std::mutex> lock(_routeMutex);
    _routeStats.clear();
}

void TransferMetrics::setEnabled(bool enabled) {
    _enabled = enabled;
}

bool TransferMetrics::isEnabled() const {
    return _enabled.load();
}

void TransferMetrics::setLogTransfers(bool log) {
    _logTransfers = log;
}

bool TransferMetrics::isLogTransfers() const {
    return _logTransfers.load();
}

void TransferMetrics::setMinBytesForLogging(uint64_t bytes) {
    _minBytesForLogging = bytes;
}

std::string TransferMetrics::makeRouteKey(int srcDevice, int dstDevice) {
    std::ostringstream oss;
    if (srcDevice < 0) oss << "host";
    else oss << "gpu" << srcDevice;
    oss << "->";
    if (dstDevice < 0) oss << "host";
    else oss << "gpu" << dstDevice;
    return oss.str();
}

void TransferMetrics::printSummary() const {
    sd_printf("%s", getSummary().c_str());
}

std::string TransferMetrics::getSummary() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    oss << "\n========== Transfer Metrics Summary ==========\n";

    auto formatRow = [&](const char* name, uint64_t count, uint64_t bytes, uint64_t timeNs) {
        double mb = bytes / (1024.0 * 1024.0);
        double ms = timeNs / 1000000.0;
        double gbps = (bytes > 0 && timeNs > 0) ? (bytes / 1e9) / (timeNs / 1e9) : 0.0;
        oss << name << ": " << count << " transfers, "
            << mb << " MB, " << ms << " ms, " << gbps << " GB/s avg\n";
    };

    formatRow("Host->Device", _h2dCount.load(), _h2dBytes.load(), _h2dTimeNs.load());
    formatRow("Device->Host", _d2hCount.load(), _d2hBytes.load(), _d2hTimeNs.load());
    formatRow("Device->Device", _d2dCount.load(), _d2dBytes.load(), _d2dTimeNs.load());
    formatRow("Peer-to-Peer", _p2pCount.load(), _p2pBytes.load(), _p2pTimeNs.load());

    oss << "\nTotal: " << getTotalBytesAllTypes() / (1024.0 * 1024.0) << " MB transferred\n";
    oss << "Total transfer time: " << getTotalTimeNsAllTypes() / 1000000.0 << " ms\n";
    oss << "Transfer overhead: " << getOverheadPercent() << "% of op time\n";

    // Route-specific stats
    if (!_routeStats.empty()) {
        oss << "\nPer-route statistics:\n";
        // Note: This is read-only so we don't need to lock
        for (const auto& pair : _routeStats) {
            double mb = pair.second.totalBytes.load() / (1024.0 * 1024.0);
            double ms = pair.second.totalTimeNs.load() / 1000000.0;
            oss << "  " << pair.first << ": " << pair.second.transferCount.load()
                << " transfers, " << mb << " MB, " << ms << " ms\n";
        }
    }

    oss << "==============================================\n";
    return oss.str();
}

}  // namespace sd
