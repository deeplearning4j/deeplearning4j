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

#pragma once

// DspSegmentHelpers — small utility functions for DSP segment execution.
//
// These are used by NativeDynamicShapePlan_cudagraph.cu for execution count
// tracking and OOM retry scheduling. Separated from DspSegmentLifecycle.h
// to keep lifecycle state machine transitions distinct from counter helpers.

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspSegmentLifecycle.h>
#include <graph/DspDiagnostics.h>

namespace sd {
namespace graph {

// Increment the segment's execution count and log the event.
static inline void dspSegIncrementExecCount(GraphSegment& seg, const char* context) {
  seg.exec.executionCount++;
  DSP_DIAG(EXECUTE, "seg[%d-%d] execCount++ -> %d (%s)",
           seg.def.startSlot, seg.def.endSlot,
           seg.exec.executionCount, context);
}

// Schedule an OOM retry for the segment using the standard retry interval.
// Delegates to SegmentLifecycle::markOomDeferred with the retry-after threshold
// computed from the current execution count and the segment's retry interval.
static inline void dspSegScheduleOomRetry(GraphSegment& seg) {
  int retryAfter = seg.exec.executionCount + GraphSegment::retryInterval();
  SegmentLifecycle::markOomDeferred(seg.exec, retryAfter);
  DSP_DIAG(MEMORY, "seg[%d-%d] OOM retry scheduled: retryAfter=%d (current=%d, interval=%d, retries=%d/%d)",
           seg.def.startSlot, seg.def.endSlot,
           retryAfter, seg.exec.executionCount,
           GraphSegment::retryInterval(),
           seg.exec.captureOomRetries, GraphSegment::maxOomRetries());
}

}  // namespace graph
}  // namespace sd
