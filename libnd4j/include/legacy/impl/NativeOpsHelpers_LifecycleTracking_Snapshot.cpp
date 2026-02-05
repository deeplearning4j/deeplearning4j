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
// Contains: Temporal snapshot and diff functions
//

#include <legacy/NativeOps.h>
#include <array/NDArrayLifecycleTracker.h>
#include <array/TADCacheLifecycleTracker.h>
#include <string>

using namespace sd::array;

/**
 * Generates a temporal leak report for NDArray allocations over time.
 */
SD_LIB_EXPORT void generateNDArrayTemporalLeakReport(const char* outputPath, int windowCount, double windowDurationSec) {
    if (outputPath == nullptr) {
        return;
    }
    std::string path(outputPath);
    NDArrayLifecycleTracker::getInstance().generateTemporalLeakReport(path, windowCount, windowDurationSec);
}

/**
 * Generates a temporal leak report for TAD cache allocations over time.
 */
SD_LIB_EXPORT void generateTADCacheTemporalLeakReport(const char* outputPath, int windowCount, double windowDurationSec) {
    if (outputPath == nullptr) {
        return;
    }
    std::string path(outputPath);
    TADCacheLifecycleTracker::getInstance().generateTemporalLeakReport(path, windowCount, windowDurationSec);
}

/**
 * Captures a snapshot of current NDArray allocations.
 */
SD_LIB_EXPORT sd::LongType captureNDArrayLeakSnapshot() {
    return NDArrayLifecycleTracker::getInstance().captureLeakSnapshot();
}

/**
 * Captures a snapshot of current TAD cache allocations.
 */
SD_LIB_EXPORT sd::LongType captureTADCacheLeakSnapshot() {
    return TADCacheLifecycleTracker::getInstance().captureLeakSnapshot();
}

/**
 * Generates a diff report between two NDArray allocation snapshots.
 */
SD_LIB_EXPORT void generateNDArraySnapshotDiff(sd::LongType snapshot1, sd::LongType snapshot2, const char* outputPath) {
    if (outputPath == nullptr) {
        return;
    }
    std::string path(outputPath);
    NDArrayLifecycleTracker::getInstance().generateSnapshotDiff(snapshot1, snapshot2, path);
}

/**
 * Generates a diff report between two TAD cache allocation snapshots.
 */
SD_LIB_EXPORT void generateTADCacheSnapshotDiff(sd::LongType snapshot1, sd::LongType snapshot2, const char* outputPath) {
    if (outputPath == nullptr) {
        return;
    }
    std::string path(outputPath);
    TADCacheLifecycleTracker::getInstance().generateSnapshotDiff(snapshot1, snapshot2, path);
}

/**
 * Clears all stored NDArray allocation snapshots to free memory.
 */
SD_LIB_EXPORT void clearNDArraySnapshots() {
    NDArrayLifecycleTracker::getInstance().clearSnapshots();
}

/**
 * Clears all stored TAD cache allocation snapshots to free memory.
 */
SD_LIB_EXPORT void clearTADCacheSnapshots() {
    TADCacheLifecycleTracker::getInstance().clearSnapshots();
}
