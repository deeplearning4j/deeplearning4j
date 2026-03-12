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

#ifndef LIBND4J_SYMBOLIC_SHAPE_RANGES_H
#define LIBND4J_SYMBOLIC_SHAPE_RANGES_H

#include <system/common.h>
#include <types/types.h>
#include <array/NDArray.h>

namespace sd {
namespace graph {

// Forward declaration — opaque data stored as void* on GraphSegment
struct SegmentShapeProfile;

/**
 * Allocate and initialize a new SegmentShapeProfile.
 * @param warmupSteps  Number of observation steps before classifying dims
 * @return Opaque pointer to store on GraphSegment::symbolicRangeData
 */
SegmentShapeProfile* createSegmentShapeProfile(int warmupSteps);

/**
 * Free a SegmentShapeProfile.
 */
void freeSegmentShapeProfile(SegmentShapeProfile* profile);

/**
 * Record observed input shapes for one execution.
 * During warmup, tracks min/max per dimension. After warmup, classifies
 * dimensions as static (never changed) or dynamic (changed at least once).
 *
 * @param profile   The shape profile for this segment
 * @param inputArrays  Array of input NDArray pointers observed this step
 * @param numInputs    Number of inputs
 */
void recordObservedShapes(SegmentShapeProfile* profile,
                          NDArray** inputArrays, int numInputs);

/**
 * Compute a range-based shape key for a segment.
 *
 * For static dimensions: hashes exact value (same as standard FNV-1a).
 * For dynamic dimensions: hashes only (rank, dtype) — the exact value
 * is a kernel arg, not baked into the compiled kernel.
 *
 * @param profile   The shape profile (must have completed warmup)
 * @param inputArrays  Current input arrays
 * @param numInputs    Number of inputs
 * @param startSlot    Segment start slot (mixed into hash)
 * @param endSlot      Segment end slot (mixed into hash)
 * @return FNV-1a hash key
 */
LongType computeRangeBasedShapeKey(SegmentShapeProfile* profile,
                                    NDArray** inputArrays, int numInputs,
                                    int startSlot, int endSlot);

/**
 * Check if warmup is complete (all observations recorded).
 */
bool isWarmupComplete(const SegmentShapeProfile* profile);

/**
 * Get the current observation count.
 */
int getObservationCount(const SegmentShapeProfile* profile);

/**
 * Get the warmup steps target.
 */
int getWarmupSteps(const SegmentShapeProfile* profile);

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_SYMBOLIC_SHAPE_RANGES_H
