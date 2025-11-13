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

// AUTO CACHE CLEANUP: Prevents cache accumulation during testing
// These caches are designed to persist in production for performance,
// but in testing mode (SD_GCC_FUNCTRACE), accumulated caches appear as leaks.
// Solution: Automatically clear caches after N operations.
namespace {
    // Operation counter for automatic cache cleanup
    std::atomic<uint64_t> g_operation_counter{0};

    // Clear caches every N operations (configurable via environment variable)
    // Default: 1 operation for testing mode (SD_GCC_FUNCTRACE)
    // This ensures caches are cleaned after every operation during leak tests
    // preventing false-positive "leaks" in lifecycle tracking reports
    // In production (without SD_GCC_FUNCTRACE), caches persist for performance
    const uint64_t DEFAULT_CLEANUP_INTERVAL = 1;

    // Get cleanup interval from environment or use default
    uint64_t getCleanupInterval() {
        static uint64_t interval = 0;
        if (interval == 0) {
            const char* env_val = std::getenv("SD_CACHE_CLEANUP_INTERVAL");
            if (env_val != nullptr) {
                interval = std::atoll(env_val);
            }
            if (interval == 0) {
                interval = DEFAULT_CLEANUP_INTERVAL;
            }
        }
        return interval;
    }

    // Check if auto-cleanup is enabled (can be disabled via environment)
    bool isAutoCleanupEnabled() {
        static int enabled = -1;
        if (enabled == -1) {
            const char* env_val = std::getenv("SD_AUTO_CACHE_CLEANUP");
            if (env_val != nullptr) {
                // "0" or "false" disables, anything else enables
                enabled = (strcmp(env_val, "0") != 0 && strcasecmp(env_val, "false") != 0) ? 1 : 0;
            } else {
                // Enabled by default in testing mode
                enabled = 1;
            }
        }
        return enabled == 1;
    }
}

// Forward declaration of cache clearing functions (defined below)
SD_LIB_EXPORT void clearTADCache();
SD_LIB_EXPORT void clearShapeCache();

/**
 * Auto-cleanup function called periodically during operation execution.
 * This is the key fix for cache accumulation during testing.
 *
 * Called from: execCustomOp2, execReduce*, execTransform*, execScalar*, etc.
 */
void checkAndCleanupCaches() {
    if (!isAutoCleanupEnabled()) {
        return;  // Auto-cleanup disabled
    }

    uint64_t count = g_operation_counter.fetch_add(1, std::memory_order_relaxed);
    uint64_t interval = getCleanupInterval();

    // Clear caches at interval boundaries
    // CRITICAL FIX: Do NOT clear shape cache or TAD cache during normal operations!
    // These caches contain buffers that are still referenced by active NDArray objects.
    // Clearing them causes use-after-free crashes when NDArrays try to access freed shape info.
    // Shape/TAD caches should only be cleared at application shutdown via explicit cleanup calls.
    if (count > 0 && (count % interval) == 0) {
        // clearTADCache();      // DISABLED - causes use-after-free
        // clearShapeCache();    // DISABLED - causes use-after-free
        // Note: We DON'T clear DifferentialFunctionClassHolder here because
        // it's a Java-side registry. The Java application should call
        // NativeOpsHolder's shutdown hook or manually call cleanup().
    }
}

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

// clearTADCache and clearShapeCache are available regardless of SD_GCC_FUNCTRACE
// because the caches themselves always exist
#include <helpers/ConstantTadHelper.h>
#include <helpers/ConstantShapeHelper.h>

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

#endif // SD_GCC_FUNCTRACE
