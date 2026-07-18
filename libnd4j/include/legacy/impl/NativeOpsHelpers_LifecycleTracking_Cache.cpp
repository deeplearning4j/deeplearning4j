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
// Contains: Cache cleanup and management functions
//

#include <legacy/NativeOps.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <atomic>
#include <cstdlib>
#include <cstring>


namespace {
    // Operation counter for automatic cache cleanup
    std::atomic<uint64_t> g_operation_counter_cache{0};

    // Get cleanup interval from environment or use default
    uint64_t getCleanupIntervalCache() {
        static uint64_t interval = 0;
        if (interval == 0) {
            const char* env_val = std::getenv("SD_CACHE_CLEANUP_INTERVAL");
            if (env_val != nullptr) {
                interval = std::atoll(env_val);
            }
            if (interval == 0) {
                interval = 100;  // Default: cleanup every 100 operations
            }
        }
        return interval;
    }

    // Check if auto-cleanup is enabled
    bool isAutoCleanupEnabledCache() {
        static int enabled = -1;
        if (enabled == -1) {
            const char* env_val = std::getenv("SD_AUTO_CACHE_CLEANUP");
            if (env_val != nullptr) {
#ifdef _WIN32
                enabled = (strcmp(env_val, "0") != 0 && _stricmp(env_val, "false") != 0) ? 1 : 0;
#else
                enabled = (strcmp(env_val, "0") != 0 && strcasecmp(env_val, "false") != 0) ? 1 : 0;
#endif
            } else {
                enabled = 0;  // DISABLED by default - cache clearing during operation causes use-after-free
            }
        }
        return enabled == 1;
    }
}

/**
 * Automatic cache cleanup called after operations.
 */
SD_LIB_EXPORT void checkAndCleanupCaches() {
    uint64_t count = g_operation_counter_cache.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t interval = getCleanupIntervalCache();

    if ((count % interval) == 0) {
        sd::ConstantTadHelper::getInstance().clearCache();
    }
}

/**
 * Clears all cached TAD packs.
 */
SD_LIB_EXPORT void clearTADCache() {
    sd::ConstantTadHelper::getInstance().clearCache();
}

/**
 * Marks that shutdown is in progress.
 */
SD_LIB_EXPORT void setTADCacheShutdownInProgress(bool inProgress) {
    sd::ConstantTadHelper::getInstance().setShutdownInProgress(inProgress);
}

/**
 * Check if TAD cache shutdown is in progress.
 */
SD_LIB_EXPORT bool isTADCacheShutdownInProgress() {
    return sd::ConstantTadHelper::getInstance().isShutdownInProgress();
}

/**
 * Marks that shutdown is in progress for the shape cache.
 * When set, clearShapeCache() becomes a no-op to avoid segfaults during JVM shutdown.
 */
SD_LIB_EXPORT void setShapeCacheShutdownInProgress(bool inProgress) {
    sd::ConstantShapeHelper::getInstance().setShutdownInProgress(inProgress);
}

/**
 * Check if shape cache shutdown is in progress.
 */
SD_LIB_EXPORT bool isShapeCacheShutdownInProgress() {
    return sd::ConstantShapeHelper::getInstance().isShutdownInProgress();
}

/**
 * Clears all cached shape buffers.
 * NOTE: Will return early without action if setShapeCacheShutdownInProgress(true) was called.
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
 * Get count of TAD cache entries (returns 0 - cache entries are not leaks).
 */
SD_LIB_EXPORT sd::LongType getTADCachedEntries() {
    return 0;
}

/**
 * Get memory used by TAD cache (returns 0 - cache memory is not leaked).
 */
SD_LIB_EXPORT sd::LongType getTADCachedBytes() {
    return 0;
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
    char* cstr = new char[result.length() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

/**
 * Get a string representation of the TAD cache for debugging.
 */
SD_LIB_EXPORT const char* getTADCacheString(int maxDepth, int maxEntries) {
    std::string result = sd::ConstantTadHelper::getInstance().toString(maxDepth, maxEntries);
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

