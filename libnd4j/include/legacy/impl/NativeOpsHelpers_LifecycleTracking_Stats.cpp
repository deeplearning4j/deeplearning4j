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
// Split from NativeOpsHelpers_LifecycleTracking.cpp to reduce object file size
// Contains: Lifecycle stats and report generation functions
//

#include <legacy/NativeOps.h>
#include <array/NDArrayLifecycleTracker.h>
#include <array/DataBufferLifecycleTracker.h>
#include <array/TADCacheLifecycleTracker.h>
#include <array/ShapeCacheLifecycleTracker.h>
#include <array/DeallocatorServiceLifecycleTracker.h>
#include <graph/OpContextLifecycleTracker.h>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <fstream>

using namespace sd::array;

/**
 * Converts NDArray lifecycle statistics to JSON format.
 */
const char* getNDArrayLifecycleStats() {
    auto stats = NDArrayLifecycleTracker::getInstance().getStats();

    std::ostringstream json;
    json << "{\n";
    json << "  \"total_allocations\": " << stats.totalAllocations << ",\n";
    json << "  \"total_deallocations\": " << stats.totalDeallocations << ",\n";
    json << "  \"current_live\": " << stats.currentLive << ",\n";
    json << "  \"total_bytes_allocated\": " << stats.totalBytesAllocated << ",\n";
    json << "  \"total_bytes_deallocated\": " << stats.totalBytesDeallocated << ",\n";
    json << "  \"peak_live\": " << stats.peakLive << "\n";
    json << "}";

    std::string result = json.str();
    char* cstr = new char[result.length() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

/**
 * Converts DataBuffer lifecycle statistics to JSON format.
 */
const char* getDataBufferLifecycleStats() {
    auto stats = DataBufferLifecycleTracker::getInstance().getStats();

    std::ostringstream json;
    json << "{\n";
    json << "  \"total_allocations\": " << stats.totalAllocations << ",\n";
    json << "  \"total_deallocations\": " << stats.totalDeallocations << ",\n";
    json << "  \"current_live\": " << stats.currentLive << ",\n";
    json << "  \"peak_live\": " << stats.peakLive << ",\n";
    json << "  \"total_bytes_allocated\": " << stats.totalBytesAllocated << ",\n";
    json << "  \"total_bytes_deallocated\": " << stats.totalBytesDeallocated << "\n";
    json << "}";

    std::string result = json.str();
    char* cstr = new char[result.length() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

/**
 * Generates a flamegraph SVG for NDArray allocations.
 */
void generateNDArrayAllocationFlamegraph(const char* outputPath) {
    if (outputPath == nullptr) {
        return;
    }

    std::string path(outputPath);
    NDArrayLifecycleTracker::getInstance().generateFlamegraph(path);
}

/**
 * Generates a flamegraph SVG for NDArray deallocations.
 */
void generateNDArrayDeallocationFlamegraph(const char* outputPath) {
    if (outputPath == nullptr) {
        return;
    }

    std::string path(outputPath);
    NDArrayLifecycleTracker::getInstance().generateDeletionFlamegraph(path);
}

/**
 * Generates a flamegraph SVG for DataBuffer allocations.
 */
void generateDataBufferAllocationFlamegraph(const char* outputPath, int bufferType) {
    if (outputPath == nullptr) {
        return;
    }

    std::string path(outputPath);
    BufferType type = (bufferType == 0) ? BufferType::PRIMARY : BufferType::SPECIAL;
    DataBufferLifecycleTracker::getInstance().generateFlamegraph(path, static_cast<int>(type));
}

/**
 * Generates a flamegraph SVG for DataBuffer deallocations.
 */
void generateDataBufferDeallocationFlamegraph(const char* outputPath, int bufferType) {
    if (outputPath == nullptr) {
        return;
    }

    std::string path(outputPath);
    BufferType type = (bufferType == 0) ? BufferType::PRIMARY : BufferType::SPECIAL;
    DataBufferLifecycleTracker::getInstance().generateDeletionFlamegraph(path, static_cast<int>(type));
}

/**
 * Generates a comprehensive leak report combining all lifecycle trackers.
 */
void generateLifecycleLeakReport(const char* outputPath) {
    if (outputPath == nullptr) {
        return;
    }

    std::string path(outputPath);

    std::ofstream combined(path);
    if (combined.is_open()) {
        combined << "============================================\n";
        combined << "  COMPREHENSIVE LIFECYCLE LEAK REPORT\n";
        combined << "============================================\n\n";

        // NDArray statistics
        auto ndarray_stats = NDArrayLifecycleTracker::getInstance().getStats();
        combined << "=== NDArray Statistics ===\n";
        combined << "  Tracking Enabled:         " << (NDArrayLifecycleTracker::getInstance().isEnabled() ? "YES" : "NO") << "\n";
        combined << "  Total Allocations:        " << ndarray_stats.totalAllocations << "\n";
        combined << "  Total Deallocations:      " << ndarray_stats.totalDeallocations << "\n";
        combined << "  Current Live:             " << ndarray_stats.currentLive << "\n";
        combined << "  Peak Live:                " << ndarray_stats.peakLive << "\n";
        combined << "  Total Bytes Allocated:    " << ndarray_stats.totalBytesAllocated << "\n";
        combined << "  Total Bytes Deallocated:  " << ndarray_stats.totalBytesDeallocated << "\n";
        combined << "\n";

        // DataBuffer statistics
        auto databuffer_stats = DataBufferLifecycleTracker::getInstance().getStats();
        combined << "=== DataBuffer Statistics ===\n";
        combined << "  Tracking Enabled:         " << (DataBufferLifecycleTracker::getInstance().isEnabled() ? "YES" : "NO") << "\n";
        combined << "  Total Allocations:        " << databuffer_stats.totalAllocations << "\n";
        combined << "  Total Deallocations:      " << databuffer_stats.totalDeallocations << "\n";
        combined << "  Current Live:             " << databuffer_stats.currentLive << "\n";
        combined << "  Peak Live:                " << databuffer_stats.peakLive << "\n";
        combined << "  Total Bytes Allocated:    " << databuffer_stats.totalBytesAllocated << "\n";
        combined << "  Total Bytes Deallocated:  " << databuffer_stats.totalBytesDeallocated << "\n";
        combined << "\n";

        // TADCache statistics
        auto tad_stats = TADCacheLifecycleTracker::getInstance().getStats();
        combined << "=== TADCache Statistics ===\n";
        combined << "  Tracking Enabled:         " << (TADCacheLifecycleTracker::getInstance().isEnabled() ? "YES" : "NO") << "\n";
        combined << "  Total Allocations:        " << tad_stats.totalAllocations << "\n";
        combined << "  Total Deallocations:      " << tad_stats.totalDeallocations << "\n";
        combined << "  Current Live:             " << tad_stats.currentLive << "\n";
        combined << "  Peak Live:                " << tad_stats.peakLive << "\n";
        combined << "  Total Bytes Allocated:    " << tad_stats.totalBytesAllocated << "\n";
        combined << "  Total Bytes Deallocated:  " << tad_stats.totalBytesDeallocated << "\n";
        combined << "\n";

        // ShapeCache statistics
        auto shape_stats = ShapeCacheLifecycleTracker::getInstance().getStats();
        combined << "=== ShapeCache Statistics ===\n";
        combined << "  Tracking Enabled:         " << (ShapeCacheLifecycleTracker::getInstance().isEnabled() ? "YES" : "NO") << "\n";
        combined << "  Total Allocations:        " << shape_stats.totalAllocations << "\n";
        combined << "  Total Deallocations:      " << shape_stats.totalDeallocations << "\n";
        combined << "  Current Live:             " << shape_stats.currentLive << "\n";
        combined << "  Peak Live:                " << shape_stats.peakLive << "\n";
        combined << "\n";

        // OpContext statistics
        auto opctx_stats = sd::graph::OpContextLifecycleTracker::getInstance().getStats();
        combined << "=== OpContext Statistics ===\n";
        combined << "  Tracking Enabled:         " << (sd::graph::OpContextLifecycleTracker::getInstance().isEnabled() ? "YES" : "NO") << "\n";
        combined << "  Total Allocations:        " << opctx_stats.totalAllocations << "\n";
        combined << "  Total Deallocations:      " << opctx_stats.totalDeallocations << "\n";
        combined << "  Current Live:             " << opctx_stats.currentLive << "\n";
        combined << "  Peak Live:                " << opctx_stats.peakLive << "\n";
        combined << "\n";

        // DeallocatorService statistics
        combined << "=== DeallocatorService Statistics ===\n";
        combined << "  Tracking Enabled:         " << (DeallocatorServiceLifecycleTracker::getInstance().isEnabled() ? "YES" : "NO") << "\n";
        combined << "  Total Allocations:        " << DeallocatorServiceLifecycleTracker::getInstance().getTotalAllocations() << "\n";
        combined << "  Total Deallocations:      " << DeallocatorServiceLifecycleTracker::getInstance().getTotalDeallocations() << "\n";
        combined << "  Current Live Count:       " << DeallocatorServiceLifecycleTracker::getInstance().getCurrentLiveCount() << "\n";
        combined << "  Current Bytes In Use:     " << DeallocatorServiceLifecycleTracker::getInstance().getCurrentBytesInUse() << "\n";
        combined << "  Peak Live Count:          " << DeallocatorServiceLifecycleTracker::getInstance().getPeakLiveCount() << "\n";
        combined << "  Peak Bytes:               " << DeallocatorServiceLifecycleTracker::getInstance().getPeakBytes() << "\n";
        combined << "\n";

        // Summary
        combined << "============================================\n";
        combined << "  SUMMARY\n";
        combined << "============================================\n";
        size_t total_leaks = ndarray_stats.currentLive + databuffer_stats.currentLive + opctx_stats.currentLive;
        if (total_leaks > 0) {
            combined << "  TOTAL POTENTIAL LEAKS: " << total_leaks << "\n";
            combined << "    - NDArrays:     " << ndarray_stats.currentLive << "\n";
            combined << "    - DataBuffers:  " << databuffer_stats.currentLive << "\n";
            combined << "    - OpContexts:   " << opctx_stats.currentLive << "\n";
        } else {
            combined << "  No leaks detected.\n";
        }
        combined << "\n";

        // Sample stack traces section
        combined << "============================================\n";
        combined << "  SAMPLE LEAK STACK TRACES\n";
        combined << "============================================\n\n";

        NDArrayLifecycleTracker::getInstance().printCurrentLeaks(combined, 5);
        combined << "\n";

        DataBufferLifecycleTracker::getInstance().printCurrentLeaks(combined, 5);
        combined << "\n";

        sd::graph::OpContextLifecycleTracker::getInstance().printCurrentLeaks(combined, 5);
        combined << "\n";

        // Per-operation analysis
        combined << "============================================\n";
        combined << "  PER-OPERATION ALLOCATION BREAKDOWN\n";
        combined << "============================================\n";

        NDArrayLifecycleTracker::getInstance().printPerOpAnalysis(combined, 3);
        combined << "\n";

        DataBufferLifecycleTracker::getInstance().printPerOpAnalysis(combined, 3);
        combined << "\n";

        sd::graph::OpContextLifecycleTracker::getInstance().printPerOpAnalysis(combined, 3);
        combined << "\n";

        // Actionable recommendations
        combined << "============================================\n";
        combined << "  ACTIONABLE RECOMMENDATIONS\n";
        combined << "============================================\n\n";

        combined << "--- TOP OPERATIONS BY LIVE ALLOCATIONS ---\n\n";

        auto ndTopOps = NDArrayLifecycleTracker::getInstance().getTopOpsByLiveCount(5);
        if (!ndTopOps.empty()) {
            combined << "  NDArray Top 5:\n";
            for (const auto& op : ndTopOps) {
                double javaPct = op.liveCount > 0 ? (100.0 * op.javaCount / op.liveCount) : 0;
                combined << "    " << op.opName << ": " << op.liveCount << " live ("
                         << (op.liveBytes / (1024*1024)) << " MB) - "
                         << javaPct << "% Java\n";
            }
            combined << "\n";
        }

        auto dbTopOps = DataBufferLifecycleTracker::getInstance().getTopOpsByLiveCount(5);
        if (!dbTopOps.empty()) {
            combined << "  DataBuffer Top 5:\n";
            for (const auto& op : dbTopOps) {
                double javaPct = op.liveCount > 0 ? (100.0 * op.javaCount / op.liveCount) : 0;
                combined << "    " << op.opName << ": " << op.liveCount << " live ("
                         << (op.liveBytes / (1024*1024)) << " MB) - "
                         << javaPct << "% Java\n";
            }
            combined << "\n";
        }

        NDArrayLifecycleTracker::getInstance().printActionableAnalysis(combined);
        combined << "\n";

        DataBufferLifecycleTracker::getInstance().printActionableAnalysis(combined);
        combined << "\n";

        // DeallocatorService status
        combined << "--- DeallocatorService Status ---\n";
        auto deallocAllocs = DeallocatorServiceLifecycleTracker::getInstance().getTotalAllocations();
        auto deallocDeallocs = DeallocatorServiceLifecycleTracker::getInstance().getTotalDeallocations();
        auto backlog = deallocAllocs - deallocDeallocs;
        double backlogPct = deallocAllocs > 0 ? (100.0 * backlog / deallocAllocs) : 0;

        combined << "  Allocations: " << deallocAllocs << "\n";
        combined << "  Deallocations: " << deallocDeallocs << "\n";
        combined << "  Backlog: " << backlog << " (" << backlogPct << "%)\n";

        if (backlogPct > 10) {
            combined << "  [WARNING] Deallocator falling behind - consider System.gc()\n";
        } else if (backlogPct > 5) {
            combined << "  [INFO] Mild deallocation lag - normal during high throughput\n";
        } else {
            combined << "  [OK] Deallocator keeping up\n";
        }
        combined << "\n";

        // Cache status
        combined << "--- Cache Actions ---\n";
        auto tadStats = TADCacheLifecycleTracker::getInstance().getStats();
        auto shapeStats = ShapeCacheLifecycleTracker::getInstance().getStats();

        combined << "  TAD Cache: " << tadStats.currentLive << " entries\n";
        combined << "  Shape Cache: " << shapeStats.currentLive << " entries\n";

        if (tadStats.currentLive > 5000) {
            combined << "  [ACTION] TAD cache large - call clearTADCache() to free memory\n";
        }

        combined << "\n";

        combined.close();
    }
}

/**
 * Generates a comprehensive leak source analysis combining ALL lifecycle trackers.
 */
void generateComprehensiveLeakAnalysis(const char* outputDir) {
    if (outputDir == nullptr) {
        return;
    }

    std::string dir(outputDir);
    std::string reportPath = dir + "/comprehensive_leak_report.txt";
    generateLifecycleLeakReport(reportPath.c_str());
}
