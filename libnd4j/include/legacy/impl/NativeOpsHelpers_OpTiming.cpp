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
// Implementation of Op Timing Tracker native API functions
//

#include <legacy/NativeOps.h>
#include <graph/profiling/OpTimingTracker.h>

using namespace sd::graph;

/**
 * Enable op timing tracker.
 * @param enabled 1 to enable, 0 to disable
 * @param detailed 1 for phase-level timing, 0 for total-only
 */
SD_LIB_EXPORT void setOpTimingEnabled(int enabled, int detailed) {
    auto& tracker = OpTimingTracker::getInstance();
    if (enabled) {
        tracker.enable(detailed != 0);
    } else {
        tracker.disable();
    }
}

/**
 * Enable op timing with Chrome trace export.
 * @param detailed 1 for phase-level timing, 0 for total-only
 */
SD_LIB_EXPORT void setOpTimingEnabledWithTrace(int detailed) {
    auto& tracker = OpTimingTracker::getInstance();
    tracker.enableWithTrace(detailed != 0);
}

/**
 * Check if op timing is enabled.
 * @return 1 if enabled, 0 if disabled
 */
SD_LIB_EXPORT int isOpTimingEnabled() {
    return OpTimingTracker::getInstance().isEnabled() ? 1 : 0;
}

/**
 * Flush ring buffer to aggregated stats.
 * Call before reading any stats.
 */
SD_LIB_EXPORT void flushOpTiming() {
    OpTimingTracker::getInstance().flush();
}

/**
 * Print top N ops by total time.
 * @param topN number of ops to show
 */
SD_LIB_EXPORT void printOpTimingStats(int topN) {
    OpTimingTracker::getInstance().printHotspots(topN);
}

/**
 * Print phase breakdown for a specific op.
 * @param opName name of the op to analyze
 */
SD_LIB_EXPORT void printOpTimingBreakdown(const char* opName) {
    if (opName != nullptr) {
        OpTimingTracker::getInstance().printOpBreakdown(std::string(opName));
    }
}

/**
 * Print timing histogram for a specific op.
 * @param opName name of the op to analyze
 */
SD_LIB_EXPORT void printOpTimingHistogram(const char* opName) {
    if (opName != nullptr) {
        OpTimingTracker::getInstance().printHistogram(std::string(opName));
    }
}

/**
 * Print per-thread timing statistics.
 */
SD_LIB_EXPORT void printOpTimingThreadStats() {
    OpTimingTracker::getInstance().printThreadStats();
}

/**
 * Reset all timing data.
 */
SD_LIB_EXPORT void resetOpTiming() {
    OpTimingTracker::getInstance().reset();
}

/**
 * Get number of unique ops tracked.
 * @return number of unique ops
 */
SD_LIB_EXPORT int getOpTimingNumOps() {
    return OpTimingTracker::getInstance().getNumOps();
}

/**
 * Get total number of op executions tracked.
 * @return total execution count
 */
SD_LIB_EXPORT sd::LongType getOpTimingTotalExecutions() {
    return static_cast<sd::LongType>(OpTimingTracker::getInstance().getTotalExecutions());
}

/**
 * Export timing data to Chrome trace JSON format.
 * @param filename output file path
 * @return 1 on success, 0 on failure
 */
SD_LIB_EXPORT int exportOpTimingChromeTrace(const char* filename) {
    if (filename == nullptr) return 0;
    return OpTimingTracker::getInstance().exportChromeTrace(std::string(filename)) ? 1 : 0;
}

/**
 * Export timing data to CSV format.
 * @param filename output file path
 * @return 1 on success, 0 on failure
 */
SD_LIB_EXPORT int exportOpTimingCSV(const char* filename) {
    if (filename == nullptr) return 0;
    return OpTimingTracker::getInstance().exportCSV(std::string(filename)) ? 1 : 0;
}
