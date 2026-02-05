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
// Contains: Lifecycle tracking enable/disable functions
//

#include <legacy/NativeOps.h>
#include <array/NDArrayLifecycleTracker.h>
#include <array/DataBufferLifecycleTracker.h>
#include <array/TADCacheLifecycleTracker.h>
#include <array/ShapeCacheLifecycleTracker.h>
#include <array/DeallocatorServiceLifecycleTracker.h>
#include <graph/OpContextLifecycleTracker.h>
#include <ops/declarable/OpExecutionLogger.h>
#include <string>

using namespace sd::array;

/**
 * Enables NDArray lifecycle tracking.
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

/**
 * Sets the current operation context for allocation tracking.
 */
SD_LIB_EXPORT void setLifecycleOpContext(const char* opName) {
    if (opName == nullptr) {
        NDArrayLifecycleTracker::clearCurrentOpContext();
        DataBufferLifecycleTracker::clearCurrentOpContext();
        sd::graph::OpContextLifecycleTracker::clearCurrentOpContext();
    } else {
        std::string op(opName);
        NDArrayLifecycleTracker::setCurrentOpContext(op);
        DataBufferLifecycleTracker::setCurrentOpContext(op);
        sd::graph::OpContextLifecycleTracker::setCurrentOpContext(op);
    }
}

/**
 * Clears the current operation context for allocation tracking.
 */
SD_LIB_EXPORT void clearLifecycleOpContext() {
    NDArrayLifecycleTracker::clearCurrentOpContext();
    DataBufferLifecycleTracker::clearCurrentOpContext();
    sd::graph::OpContextLifecycleTracker::clearCurrentOpContext();
}

/**
 * Gets the current operation context for allocation tracking.
 */
SD_LIB_EXPORT const char* getLifecycleOpContext() {
    thread_local static std::string g_opContext;
    g_opContext = NDArrayLifecycleTracker::getCurrentOpContext();
    return g_opContext.c_str();
}

/**
 * Set the current allocation context (operation name) for lifecycle tracking.
 */
SD_LIB_EXPORT void setAllocationContext(const char* opName) {
    if (opName != nullptr) {
        std::string op(opName);
        sd::ops::OpExecutionLogger::setCurrentOpName(op);
        NDArrayLifecycleTracker::setCurrentOpContext(op);
        DataBufferLifecycleTracker::setCurrentOpContext(op);
        sd::graph::OpContextLifecycleTracker::setCurrentOpContext(op);
    }
}

/**
 * Clear the current allocation context for this thread.
 */
SD_LIB_EXPORT void clearAllocationContext() {
    sd::ops::OpExecutionLogger::clearCurrentOpName();
    NDArrayLifecycleTracker::clearCurrentOpContext();
    DataBufferLifecycleTracker::clearCurrentOpContext();
    sd::graph::OpContextLifecycleTracker::clearCurrentOpContext();
}

// DeallocatorService Lifecycle Tracking

/**
 * Record a snapshot of DeallocatorService statistics from Java.
 */
SD_LIB_EXPORT void recordDeallocatorServiceSnapshot(
    sd::LongType totalAllocations, sd::LongType totalDeallocations,
    sd::LongType totalBytesAllocated, sd::LongType totalBytesDeallocated,
    sd::LongType peakLiveCount, sd::LongType peakBytes) {

    DeallocatorServiceLifecycleTracker::getInstance().recordSnapshot(
        static_cast<uint64_t>(totalAllocations),
        static_cast<uint64_t>(totalDeallocations),
        static_cast<uint64_t>(totalBytesAllocated),
        static_cast<uint64_t>(totalBytesDeallocated),
        static_cast<uint64_t>(peakLiveCount),
        static_cast<uint64_t>(peakBytes)
    );
}

/**
 * Enable DeallocatorService lifecycle tracking.
 */
SD_LIB_EXPORT void enableDeallocatorServiceTracking() {
    DeallocatorServiceLifecycleTracker::getInstance().enable();
}

/**
 * Disable DeallocatorService lifecycle tracking.
 */
SD_LIB_EXPORT void disableDeallocatorServiceTracking() {
    DeallocatorServiceLifecycleTracker::getInstance().disable();
}

/**
 * Check if DeallocatorService tracking is enabled.
 */
SD_LIB_EXPORT bool isDeallocatorServiceTrackingEnabled() {
    return DeallocatorServiceLifecycleTracker::getInstance().isEnabled();
}

/**
 * Get current live count from DeallocatorService tracker.
 */
SD_LIB_EXPORT sd::LongType getDeallocatorServiceLiveCount() {
    return static_cast<sd::LongType>(
        DeallocatorServiceLifecycleTracker::getInstance().getCurrentLiveCount());
}

/**
 * Get current bytes in use from DeallocatorService tracker.
 */
SD_LIB_EXPORT sd::LongType getDeallocatorServiceBytesInUse() {
    return static_cast<sd::LongType>(
        DeallocatorServiceLifecycleTracker::getInstance().getCurrentBytesInUse());
}
