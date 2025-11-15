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
// Implementation of lifecycle tracking native API functions
// Author: Adam Gibson
//

#include <legacy/NativeOps.h>

#if defined(SD_GCC_FUNCTRACE)

// Forward declare the ComprehensiveLeakAnalyzer class before including tracker headers
// This ensures the friend declarations in the tracker classes can see the class
namespace sd {
namespace analysis {
    class ComprehensiveLeakAnalyzer;
}
}

#include <array/NDArrayLifecycleTracker.h>
#include <array/DataBufferLifecycleTracker.h>
#include <array/TADCacheLifecycleTracker.h>
#include <array/ShapeCacheLifecycleTracker.h>
#include <graph/OpContextLifecycleTracker.h>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <atomic>
#include <cstdlib>

using namespace sd::array;

// AUTO CACHE CLEANUP: Moved outside #if defined(SD_GCC_FUNCTRACE) guard
// NOTE: The duplicate namespace block with g_operation_counter and helper functions
// has been removed. The implementation now uses the "_nocache" versions defined
// outside the SD_GCC_FUNCTRACE guard (see around line 424) to ensure cleanup works
// in all builds, not just functrace builds.

// NOTE: checkAndCleanupCaches() has been moved OUTSIDE the #if defined(SD_GCC_FUNCTRACE) guard
// to ensure it's available in all builds (see implementation around line 478).
// Previous duplicate implementation here has been removed to fix linker symbol conflict.
//
// Forward declarations for cache clearing functions (implementations are outside this guard)
SD_LIB_EXPORT void clearTADCache();
SD_LIB_EXPORT void clearShapeCache();
SD_LIB_EXPORT void checkAndCleanupCaches();

// Include comprehensive leak analysis implementation
#include "../generate_leak_analysis.cpp"

/**
 * Converts NDArray lifecycle statistics to JSON format.
 */
const char* getNDArrayLifecycleStats() {
    auto stats = NDArrayLifecycleTracker::getInstance().getStats();

    std::ostringstream json;
    json << "{\n";
    json << "  \"total_allocations\": " << stats.total_allocs << ",\n";
    json << "  \"total_deallocations\": " << stats.total_deallocs << ",\n";
    json << "  \"current_live\": " << stats.current_live << ",\n";
    json << "  \"current_bytes\": " << stats.current_bytes << ",\n";
    json << "  \"peak_bytes\": " << stats.peak_bytes << ",\n";
    json << "  \"double_frees\": " << stats.double_frees << "\n";
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
    json << "  \"primary\": {\n";
    json << "    \"total_allocations\": " << stats.total_primary_allocs << ",\n";
    json << "    \"total_deallocations\": " << stats.total_primary_deallocs << ",\n";
    json << "    \"current_live\": " << stats.current_live_primary << ",\n";
    json << "    \"current_bytes\": " << stats.current_primary_bytes << ",\n";
    json << "    \"peak_bytes\": " << stats.peak_primary_bytes << "\n";
    json << "  },\n";
    json << "  \"special\": {\n";
    json << "    \"total_allocations\": " << stats.total_special_allocs << ",\n";
    json << "    \"total_deallocations\": " << stats.total_special_deallocs << ",\n";
    json << "    \"current_live\": " << stats.current_live_special << ",\n";
    json << "    \"current_bytes\": " << stats.current_special_bytes << ",\n";
    json << "    \"peak_bytes\": " << stats.peak_special_bytes << "\n";
    json << "  },\n";
    json << "  \"double_frees\": " << stats.double_frees << "\n";
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
    DataBufferLifecycleTracker::getInstance().generateFlamegraph(path, type);
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
    DataBufferLifecycleTracker::getInstance().generateDeletionFlamegraph(path, type);
}

/**
 * Generates a comprehensive leak report combining NDArray and DataBuffer leaks.
 */
void generateLifecycleLeakReport(const char* outputPath) {
    if (outputPath == nullptr) {
        return;
    }

    std::string path(outputPath);

    // Generate separate reports then combine them
    std::string ndarray_report = path + ".ndarray.txt";
    std::string databuffer_report = path + ".databuffer.txt";

    // Generate individual reports
    NDArrayLifecycleTracker::getInstance().generateLeakReport(ndarray_report);
    DataBufferLifecycleTracker::getInstance().generateLeakReport(databuffer_report);

    // Create combined report
    std::ofstream combined(path);
    if (combined.is_open()) {
        combined << "============================================\n";
        combined << "  COMBINED LIFECYCLE LEAK REPORT\n";
        combined << "============================================\n\n";

        // NDArray statistics
        auto ndarray_stats = NDArrayLifecycleTracker::getInstance().getStats();
        combined << "NDArray Statistics:\n";
        combined << "  Total Allocations:   " << ndarray_stats.total_allocs << "\n";
        combined << "  Total Deallocations: " << ndarray_stats.total_deallocs << "\n";
        combined << "  Current Live:        " << ndarray_stats.current_live << "\n";
        combined << "  Current Bytes:       " << ndarray_stats.current_bytes << "\n";
        combined << "  Peak Bytes:          " << ndarray_stats.peak_bytes << "\n";
        combined << "  Double Frees:        " << ndarray_stats.double_frees << "\n\n";

        // DataBuffer statistics
        auto databuffer_stats = DataBufferLifecycleTracker::getInstance().getStats();
        combined << "DataBuffer Statistics:\n";
        combined << "  PRIMARY (Host) Memory:\n";
        combined << "    Total Allocations:   " << databuffer_stats.total_primary_allocs << "\n";
        combined << "    Total Deallocations: " << databuffer_stats.total_primary_deallocs << "\n";
        combined << "    Current Live:        " << databuffer_stats.current_live_primary << "\n";
        combined << "    Current Bytes:       " << databuffer_stats.current_primary_bytes << "\n";
        combined << "    Peak Bytes:          " << databuffer_stats.peak_primary_bytes << "\n";
        combined << "  SPECIAL (Device) Memory:\n";
        combined << "    Total Allocations:   " << databuffer_stats.total_special_allocs << "\n";
        combined << "    Total Deallocations: " << databuffer_stats.total_special_deallocs << "\n";
        combined << "    Current Live:        " << databuffer_stats.current_live_special << "\n";
        combined << "    Current Bytes:       " << databuffer_stats.current_special_bytes << "\n";
        combined << "    Peak Bytes:          " << databuffer_stats.peak_special_bytes << "\n";
        combined << "  Double Frees:          " << databuffer_stats.double_frees << "\n\n";

        combined << "See detailed reports:\n";
        combined << "  " << ndarray_report << "\n";
        combined << "  " << databuffer_report << "\n";

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
    runComprehensiveLeakAnalysis(dir.c_str());
}

#endif // SD_GCC_FUNCTRACE

// Enable/disable functions are ALWAYS available, regardless of SD_GCC_FUNCTRACE
// They will simply enable/disable tracking if the trackers are compiled in

#if defined(SD_GCC_FUNCTRACE)

/**
 * Enables NDArray lifecycle tracking.
 * When enabled, all NDArray allocations and deallocations are tracked
 * with stack traces for leak detection.
 */
SD_LIB_EXPORT void enableNDArrayTracking() {
    NDArrayLifecycleTracker::getInstance().setEnabled(true);
}

/**
 * Disables NDArray lifecycle tracking.
 */
SD_LIB_EXPORT void disableNDArrayTracking() {
    NDArrayLifecycleTracker::getInstance().setEnabled(false);
}

/**
 * Enables DataBuffer lifecycle tracking.
 * When enabled, all DataBuffer allocations and deallocations are tracked
 * with stack traces for leak detection.
 */
SD_LIB_EXPORT void enableDataBufferTracking() {
    DataBufferLifecycleTracker::getInstance().setEnabled(true);
}

/**
 * Disables DataBuffer lifecycle tracking.
 */
SD_LIB_EXPORT void disableDataBufferTracking() {
    DataBufferLifecycleTracker::getInstance().setEnabled(false);
}

/**
 * Enables TADCache lifecycle tracking.
 * When enabled, all TAD (Tensor Along Dimension) cache allocations and deallocations
 * are tracked with stack traces for leak detection.
 */
SD_LIB_EXPORT void enableTADCacheTracking() {
    TADCacheLifecycleTracker::getInstance().setEnabled(true);
}

/**
 * Disables TADCache lifecycle tracking.
 */
SD_LIB_EXPORT void disableTADCacheTracking() {
    TADCacheLifecycleTracker::getInstance().setEnabled(false);
}

/**
 * Enables ShapeCache lifecycle tracking.
 * When enabled, all shape cache allocations and deallocations are tracked
 * with stack traces for leak detection.
 */
SD_LIB_EXPORT void enableShapeCacheTracking() {
    ShapeCacheLifecycleTracker::getInstance().setEnabled(true);
}

/**
 * Disables ShapeCache lifecycle tracking.
 */
SD_LIB_EXPORT void disableShapeCacheTracking() {
    ShapeCacheLifecycleTracker::getInstance().setEnabled(false);
}

/**
 * Enables OpContext lifecycle tracking.
 * When enabled, all operation context allocations and deallocations are tracked
 * with stack traces for leak detection.
 */
SD_LIB_EXPORT void enableOpContextTracking() {
    sd::graph::OpContextLifecycleTracker::getInstance().setEnabled(true);
}

/**
 * Disables OpContext lifecycle tracking.
 */
SD_LIB_EXPORT void disableOpContextTracking() {
    sd::graph::OpContextLifecycleTracker::getInstance().setEnabled(false);
}

#endif // SD_GCC_FUNCTRACE

// AUTO CACHE CLEANUP - MOVED OUTSIDE SD_GCC_FUNCTRACE GUARD
// Critical fix: Cache cleanup must work even without functrace!
// The caches accumulate regardless of tracking, so cleanup must always be available.
#include <helpers/ConstantTadHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace {
    // Operation counter for automatic cache cleanup
    std::atomic<uint64_t> g_operation_counter_nocache{0};

    // Get cleanup interval from environment or use default
    uint64_t getCleanupIntervalNoCache() {
        static uint64_t interval = 0;
        if (interval == 0) {
            const char* env_val = std::getenv("SD_CACHE_CLEANUP_INTERVAL");
            if (env_val != nullptr) {
                interval = std::atoll(env_val);
            }
            if (interval == 0) {
                // Use 10 for all cases (reduced from 100 to fix 135 MB TAD cache leak)
                // Previous interval=100 was too large for leak tests (~130 operations)
                // causing packs from operations 101-130 to remain in cache
                // interval=10 ensures cleanup happens frequently enough for testing
                // while still being conservative for production use
                interval = 10;
            }
        }
        return interval;
    }

    // Check if auto-cleanup is enabled
    bool isAutoCleanupEnabledNoCache() {
        static int enabled = -1;
        if (enabled == -1) {
            const char* env_val = std::getenv("SD_AUTO_CACHE_CLEANUP");
            if (env_val != nullptr) {
                enabled = (strcmp(env_val, "0") != 0 && strcasecmp(env_val, "false") != 0) ? 1 : 0;
            } else {
                // Enabled by default
                enabled = 1;
            }
        }
        return enabled == 1;
    }
}

// Forward declarations for cache clearing
SD_LIB_EXPORT void clearTADCache();
SD_LIB_EXPORT void clearShapeCache();

/**
 * Auto-cleanup function - NOW ALWAYS AVAILABLE (not just with SD_GCC_FUNCTRACE)
 *
 * CRITICAL FIX FOR 135 MB TAD CACHE LEAK:
 * Previous implementation was inside #if defined(SD_GCC_FUNCTRACE) block,
 * which meant cleanup only worked in functrace builds. TAD caches accumulate
 * in ALL builds, so cleanup must be available regardless of tracking.
 *
 * Called from: execCustomOp2, execReduce*, execTransform*, execScalar*, etc.
 */
SD_LIB_EXPORT void checkAndCleanupCaches() {
    if (!isAutoCleanupEnabledNoCache()) {
        return;  // Auto-cleanup disabled
    }

    // Post-increment: get new value AFTER incrementing
    // This ensures cleanup happens at operations 10, 20, 30, etc. (not 11, 21, 31)
    uint64_t count = g_operation_counter_nocache.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t interval = getCleanupIntervalNoCache();

    // Clear caches at interval boundaries
    if ((count % interval) == 0) {
        // CRITICAL: Use fprintf to stderr for unconditional logging
        // sd_printf may be disabled/redirected, but stderr always works
        fprintf(stderr, "[TADCache] Operation %llu: Triggering cleanup (interval=%llu)\n",
                (unsigned long long)count, (unsigned long long)interval);
        fflush(stderr);

        clearTADCache();        // ENABLED - clears TAD cache every N operations

        fprintf(stderr, "[TADCache] Operation %llu: Cleanup completed\n",
                (unsigned long long)count);
        fflush(stderr);
        // clearShapeCache();   // KEEP DISABLED - shape info has wider usage than TAD
    }
}

/**
 * Clears all cached TAD packs to prevent memory leaks during testing.
 * This is particularly useful when running memory leak tests that
 * track allocations, as it allows clearing accumulated cache between tests.
 */
SD_LIB_EXPORT void clearTADCache() {
    sd::ConstantTadHelper::getInstance().clearCache();
}

/**
 * Clears all cached shape buffers to prevent memory leaks.
 * This is called during application shutdown to free accumulated cache memory.
 */
SD_LIB_EXPORT void clearShapeCache() {
    sd::ConstantShapeHelper::getInstance().clearCache();
}

/**
 * Get the total number of cached shape buffer entries.
 */
SD_LIB_EXPORT sd::LongType getShapeCachedEntries() {
    return sd::ConstantShapeHelper::getInstance().getCachedEntries();
}

/**
 * Get the total memory used by cached shape buffers in bytes.
 */
SD_LIB_EXPORT sd::LongType getShapeCachedBytes() {
    return sd::ConstantShapeHelper::getInstance().getCachedBytes();
}

/**
 * Get the peak number of shape entries that were cached simultaneously.
 */
SD_LIB_EXPORT sd::LongType getShapePeakCachedEntries() {
    return sd::ConstantShapeHelper::getInstance().getPeakCachedEntries();
}

/**
 * Get the peak memory usage by cached shape buffers in bytes.
 */
SD_LIB_EXPORT sd::LongType getShapePeakCachedBytes() {
    return sd::ConstantShapeHelper::getInstance().getPeakCachedBytes();
}

/**
 * Get the total number of cached TAD pack entries.
 */
SD_LIB_EXPORT sd::LongType getTADCachedEntries() {
    return sd::ConstantTadHelper::getInstance().getCachedEntries();
}

/**
 * Get the total memory used by cached TAD packs in bytes.
 */
SD_LIB_EXPORT sd::LongType getTADCachedBytes() {
    return sd::ConstantTadHelper::getInstance().getCachedBytes();
}

/**
 * Get the peak number of TAD pack entries that were cached simultaneously.
 */
SD_LIB_EXPORT sd::LongType getTADPeakCachedEntries() {
    return sd::ConstantTadHelper::getInstance().getPeakCachedEntries();
}

/**
 * Get the peak memory usage by cached TAD packs in bytes.
 */
SD_LIB_EXPORT sd::LongType getTADPeakCachedBytes() {
    return sd::ConstantTadHelper::getInstance().getPeakCachedBytes();
}

/**
 * Get a string representation of the shape cache for debugging.
 */
SD_LIB_EXPORT const char* getShapeCacheString(int maxDepth, int maxEntries) {
    std::string result = sd::ConstantShapeHelper::getInstance().toString(maxDepth, maxEntries);
    // Allocate C-style string that Java can read
    char* cstr = new char[result.length() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

/**
 * Get a string representation of the TAD cache for debugging.
 */
SD_LIB_EXPORT const char* getTADCacheString(int maxDepth, int maxEntries) {
    std::string result = sd::ConstantTadHelper::getInstance().toString(maxDepth, maxEntries);
    // Allocate C-style string that Java can read
    char* cstr = new char[result.length() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

/**
 * Free a string returned by native code.
 */
SD_LIB_EXPORT void freeString(const char* ptr) {
    if (ptr != nullptr) {
        delete[] ptr;
    }
}

#if !defined(SD_GCC_FUNCTRACE)
// When SD_GCC_FUNCTRACE is not defined - lifecycle tracking disabled

// When SD_GCC_FUNCTRACE is not defined, trackers don't exist, so these are no-ops
SD_LIB_EXPORT void enableNDArrayTracking() {}
SD_LIB_EXPORT void disableNDArrayTracking() {}
SD_LIB_EXPORT void enableDataBufferTracking() {}
SD_LIB_EXPORT void disableDataBufferTracking() {}
SD_LIB_EXPORT void enableTADCacheTracking() {}
SD_LIB_EXPORT void disableTADCacheTracking() {}
SD_LIB_EXPORT void enableShapeCacheTracking() {}
SD_LIB_EXPORT void disableShapeCacheTracking() {}
SD_LIB_EXPORT void enableOpContextTracking() {}
SD_LIB_EXPORT void disableOpContextTracking() {}

// Stub implementations for lifecycle query and generation functions
// These return empty/null values when tracking is not compiled in
// NOTE: checkAndCleanupCaches() is now always available (moved outside #ifdef block)
// to ensure TAD cache cleanup works in all builds, not just functrace builds.

SD_LIB_EXPORT const char* getNDArrayLifecycleStats() {
    // Return empty JSON object when tracking is disabled
    static const char* empty_stats = "{}";
    return empty_stats;
}

SD_LIB_EXPORT const char* getDataBufferLifecycleStats() {
    // Return empty JSON object when tracking is disabled
    static const char* empty_stats = "{}";
    return empty_stats;
}

SD_LIB_EXPORT void generateNDArrayAllocationFlamegraph(const char* outputPath) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateNDArrayDeallocationFlamegraph(const char* outputPath) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateDataBufferAllocationFlamegraph(const char* outputPath, int bufferType) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateDataBufferDeallocationFlamegraph(const char* outputPath, int bufferType) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateLifecycleLeakReport(const char* outputPath) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateComprehensiveLeakAnalysis(const char* outputDir) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateNDArrayTemporalLeakReport(const char* outputPath, int windowCount, double windowDurationSec) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateTADCacheTemporalLeakReport(const char* outputPath, int windowCount, double windowDurationSec) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT sd::LongType captureNDArrayLeakSnapshot() {
    return 0;
}

SD_LIB_EXPORT sd::LongType captureTADCacheLeakSnapshot() {
    return 0;
}

SD_LIB_EXPORT void generateNDArraySnapshotDiff(sd::LongType snapshot1, sd::LongType snapshot2, const char* outputPath) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateTADCacheSnapshotDiff(sd::LongType snapshot1, sd::LongType snapshot2, const char* outputPath) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void clearNDArraySnapshots() {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void clearTADCacheSnapshots() {
    // No-op when tracking is disabled
}

#endif // SD_GCC_FUNCTRACE
