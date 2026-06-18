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

#ifndef LIBND4J_GRAPH_ANALYSIS_UTILS_H
#define LIBND4J_GRAPH_ANALYSIS_UTILS_H

#include <config.h>

#if defined(SD_CUDA) || defined(HAVE_MLIR) || HAVE_TRITON || HAVE_MLX

#include <graph/NativeDynamicShapePlan.h>
#include <graph/SegmentAnalysisTypes.h>

#include <unordered_set>

namespace sd {
namespace graph {

/**
 * Shared segment analysis utilities used by both GPU (Triton) and CPU (MLIR) backends.
 * Eliminates code duplication between CpuIRBuilder and TritonIRBuilder.
 *
 * These methods perform pure analysis over NativeSlot arrays — no MLIR, Triton,
 * or backend-specific dependencies.
 */
class GraphAnalysisUtils {
 public:
  /**
   * Profile a segment: build dataflow graph, count op categories, resolve shapes.
   * Used by both CpuIRBuilder and TritonIRBuilder.
   *
   * Walks [startSlot, endSlot] in a single O(n) pass, building an OpNode per slot
   * with dataflow edges, category classification, and output shape resolution.
   */
  static SegmentProfile profileSegment(NativeSlot* slots, int startSlot, int endSlot,
                                        NDArray** outputSlots, int totalOutputSlots);

  /**
   * Compute the set of output slot indices that are consumed by ops AFTER endSlot
   * (or are produced by the last slot in the segment).
   * These outputs must be materialized to global memory.
   */
  static std::unordered_set<int> computeExternallyVisibleOutputs(
      NativeSlot* slots, int startSlot, int endSlot, int totalSlots);
};

}  // namespace graph
}  // namespace sd

#endif  // defined(SD_CUDA) || defined(HAVE_MLIR) || HAVE_TRITON || HAVE_MLX
#endif  // LIBND4J_GRAPH_ANALYSIS_UTILS_H
