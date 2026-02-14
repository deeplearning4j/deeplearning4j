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

#ifndef LIBND4J_NATIVE_DYNAMIC_SHAPE_PLAN_H
#define LIBND4J_NATIVE_DYNAMIC_SHAPE_PLAN_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <graph/generated/graph_generated.h>
#include <ops/declarable/DeclarableOp.h>
#include <system/common.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef SD_CUDA
#include <execution/cuda/CudaGraphScheduler.h>
#endif

// Forward declare graph backends (included conditionally in .cpp)
namespace sd { namespace graph { class GraphBackend; } }

namespace sd {
namespace graph {

// FlatGraph is in the ::graph namespace (FlatBuffer-generated)

/**
 * Source type constants for inputSourceTypes, mirroring DynamicShapeSlot.java.
 */
enum NativeSourceType : int8_t {
  SOURCE_CONSTANT = 0,
  SOURCE_VARIABLE = 1,
  SOURCE_PLACEHOLDER = 2,
  SOURCE_OP_OUTPUT = 3,
};

/**
 * Per-op descriptor with pre-compiled wiring for the native plan executor.
 * Mirrors DynamicShapeSlot.java but uses C++ types.
 *
 * Index conventions for inputSourceIndices:
 *   >= 0: index into the flat outputSlots array (from a prior op's output)
 *   <  0: index into external arrays: -(index + 1) into the external input array
 */
struct NativeSlot {
  // Op identification
  LongType opHash;                         // Op hash for lookup
  sd::ops::DeclarableOp* op;       // Resolved at compile time (not owned)
  std::string opName;                      // For diagnostics

  // Input wiring
  int numInputs;
  int* inputSourceIndices;                 // >=0: prior slot, <0: external (-(idx+1))
  int8_t* inputSourceTypes;               // NativeSourceType values

  // Output wiring
  int numOutputs;
  int* outputSlotIndices;                  // Flat slot indices for each output

  // Frozen op arguments
  LongType* iArgs;    int numIArgs;
  double* tArgs;      int numTArgs;
  bool* bArgs;        int numBArgs;
  DataType* dArgs;    int numDArgs;

  // Execution flags
  bool needsZeroedOutput;
  bool isDataDependent;
  bool outputShapeDependsOnInputValues;
  bool needsIntLongSync;
  bool isCustomOp;
  int targetDeviceId;                      // -1 = auto

  // Legacy op support for ops not registered in OpRegistrator
  // (exp, log, abs, neg, sqrt, sin, cos, etc.)
  // legacyOpType: 0=not legacy, 1=TransformSame, 2=TransformStrict,
  //               3=TransformFloat, 4=TransformBool, 5=Scalar, 6=PairwiseTransform
  int legacyOpType;
  int legacyOpNum;

  // Per-slot shape cache
  LongType cachedShapeKey;
  std::vector<const LongType*> cachedOutputShapes;   // Cached shape infos (not owned)
  bool shapeCacheValid;

  NativeSlot()
      : opHash(0), op(nullptr), numInputs(0), inputSourceIndices(nullptr),
        inputSourceTypes(nullptr), numOutputs(0), outputSlotIndices(nullptr),
        iArgs(nullptr), numIArgs(0), tArgs(nullptr), numTArgs(0),
        bArgs(nullptr), numBArgs(0), dArgs(nullptr), numDArgs(0),
        needsZeroedOutput(true), isDataDependent(false),
        outputShapeDependsOnInputValues(false), needsIntLongSync(false),
        isCustomOp(true), targetDeviceId(-1),
        legacyOpType(0), legacyOpNum(-1),
        cachedShapeKey(0), shapeCacheValid(false) {}

  ~NativeSlot() {
    delete[] inputSourceIndices;
    delete[] inputSourceTypes;
    delete[] outputSlotIndices;
    delete[] iArgs;
    delete[] tArgs;
    delete[] bArgs;
    delete[] dArgs;
  }

  // No copy
  NativeSlot(const NativeSlot&) = delete;
  NativeSlot& operator=(const NativeSlot&) = delete;

  // Move OK
  NativeSlot(NativeSlot&& other) noexcept;
  NativeSlot& operator=(NativeSlot&& other) noexcept;
};

/**
 * Graph segment for CUDA Graph capture.
 * A contiguous range of slots that can be captured as a single CUDA graph.
 *
 * Lifecycle:
 *   executionCount == 0: warm-up pass (slot-by-slot, populates slot cache)
 *   executionCount == 1: capture pass (ops recorded into CUDA graph, then launched)
 *   executionCount >= 2: replay pass (cached graph launched directly)
 */
struct GraphSegment {
  int startSlot;
  int endSlot;
  bool isCapturable;

  // Shape key for cache invalidation
  LongType shapeKey;

  // Execution tracking
  int executionCount;

#ifdef SD_CUDA
  // Cached CUDA graph for replay
  std::shared_ptr<sd::cuda::CudaGraphHandle> cachedGraph;
  LongType cachedShapeKey;
  bool captureFailed;  // If true, never attempt capture for this segment
#endif

  GraphSegment()
      : startSlot(0), endSlot(0), isCapturable(false), shapeKey(0),
        executionCount(0)
#ifdef SD_CUDA
        , cachedShapeKey(0), captureFailed(false)
#endif
  {}
};

/**
 * Native C++ plan executor that replaces the Java DynamicShapePlanExecutor.
 *
 * Executes the entire pre-compiled plan in C++ with a single JNI call,
 * eliminating per-op Java→JNI→C++ round-trip overhead (~15-20μs per op).
 *
 * Two construction paths:
 * 1. fromSerializedPlan() - from serialized binary plan (sent from Java via JNI)
 * 2. fromFlatGraph() - from FlatGraph + variables (C++-native loading via SdzReader)
 */
class SD_LIB_EXPORT NativeDynamicShapePlan {
 public:
  /**
   * Construct from a serialized binary plan (sent from Java).
   *
   * Binary format:
   *   Header: magic("DSP1") + version(int32) + numSlots(int32) + totalOutputSlots(int32)
   *           + numExternalInputs(int32) + numRequestedOutputs(int32)
   *   Per-slot: opHash(int64), numInputs(int32), numOutputs(int32),
   *             inputSourceIndices[numInputs](int32),
   *             inputSourceTypes[numInputs](int8),
   *             outputSlotIndices[numOutputs](int32),
   *             numIArgs/numTArgs/numBArgs/numDArgs(int32 each),
   *             iArgs[](int64), tArgs[](double), bArgs[](bool), dArgs[](int32),
   *             flags: needsZeroedOutput(bool), isDataDependent(bool),
   *                    outputShapeDependsOnInputValues(bool), needsIntLongSync(bool),
   *                    isCustomOp(bool), targetDeviceId(int32)
   *   Release schedule: for each step: count(int32) + slotIndices[count](int32)
   *   Requested outputs: for each output: slotIndex(int32)
   */
  static NativeDynamicShapePlan* fromSerializedPlan(const void* data, LongType size);

  /**
   * Construct from a FlatGraph + pre-loaded variables.
   * Used for pure C++ model loading (no Java needed).
   */
  static NativeDynamicShapePlan* fromFlatGraph(
      const ::graph::FlatGraph* graph,
      const std::unordered_map<std::string, NDArray*>& variables,
      const std::vector<std::string>& requestedOutputs);

  ~NativeDynamicShapePlan();

  // No copy
  NativeDynamicShapePlan(const NativeDynamicShapePlan&) = delete;
  NativeDynamicShapePlan& operator=(const NativeDynamicShapePlan&) = delete;

  /**
   * Execute the full plan.
   *
   * @param externalInputs  Array of NDArray* for constants/variables/placeholders
   * @param numExternalInputs  Number of external inputs
   * @param requestedOutputs  Pre-allocated array to receive output NDArrays (caller owns)
   * @param numRequestedOutputs  Number of requested outputs
   * @param stream  CUDA stream (nullptr for CPU)
   * @return Status::OK on success
   */
  Status execute(
      NDArray** externalInputs, int numExternalInputs,
      NDArray** requestedOutputs, int numRequestedOutputs,
      void* stream);

  /**
   * Clear all per-slot shape caches.
   * Must be called when session resets to avoid stale GPU memory references.
   */
  void clearShapeCaches();

  /**
   * Get the number of external inputs expected by this plan.
   */
  int getNumExternalInputs() const { return numExternalInputs_; }

  /**
   * Get the number of requested outputs.
   */
  int getNumRequestedOutputs() const { return numRequestedOutputs_; }

  /**
   * Get the total number of slots (ops) in the plan.
   */
  int getNumSlots() const { return numSlots_; }

  /**
   * Get the total number of output slots (intermediate + final).
   */
  int getTotalOutputSlots() const { return totalOutputSlots_; }

  /**
   * Get plan segments (for CUDA Graphs integration).
   */
  const std::vector<GraphSegment>& getSegments() const { return segments_; }

  /**
   * Get CUDA Graph execution statistics.
   * Returns number of segments captured as CUDA graphs.
   */
  int getNumCapturedGraphSegments() const;

  /**
   * Get total number of CUDA graph replays across all segments.
   */
  int getTotalGraphReplays() const;

  /**
   * Enable/disable CUDA Graphs for this plan.
   * Default: disabled (slot-by-slot execution).
   */
  void setCudaGraphsEnabled(bool enabled) { cudaGraphsEnabled_ = enabled; }
  bool isCudaGraphsEnabled() const { return cudaGraphsEnabled_; }

  friend class NativePlanCompiler;

 private:
  NativeDynamicShapePlan();

  // Slot data
  NativeSlot* slots_;
  int numSlots_;
  int totalOutputSlots_;
  int numExternalInputs_;

  // Release schedule: releaseAtStep_[stepIdx] = array of slot indices to release
  int** releaseAtStep_;
  int* releaseAtStepCounts_;

  // Requested output mapping
  int* requestedOutputSlotIndices_;
  int numRequestedOutputs_;

  // Execution state (reused across calls)
  NDArray** outputSlots_;              // Current output slot values
  NDArray** slotArrayCache_;           // Per-slot cached arrays for reuse
  bool* slotIsViewProducer_;           // View producer flags (learned from first exec)
  Context** contextPool_;              // Pre-allocated Context pool
  bool viewProducerDetectionDone_;

  // Graph segments for CUDA Graphs
  std::vector<GraphSegment> segments_;

  // Memory management: evicted NDArrays awaiting deletion
  std::vector<NDArray*> pendingClose_;
  size_t pendingCloseBytes_;

  // CUDA Graphs control
  bool cudaGraphsEnabled_;
  int totalGraphReplays_;

  // Owned legacy ops created during deserialization
  // (for ops not registered in OpRegistrator)
  std::vector<sd::ops::DeclarableOp*> ownedLegacyOps_;

  // Internal methods
  Status executeSlot(int slotIdx, NDArray** externalArrays, int numExt, void* stream);
  LongType computeShapeKey(NativeSlot& slot, NDArray** inputs, int numInputs);
  LongType computeSegmentShapeKey(GraphSegment& seg, NDArray** externalInputs, int numExt);
  void flushPendingClose(void* stream);
  void buildSegments();

  // Segment execution strategies
  Status executeSegmentSlotBySlot(GraphSegment& seg, NDArray** externalArrays,
                                  int numExt, void* stream);
#ifdef SD_CUDA
  Status executeSegmentWithGraph(GraphSegment& seg, NDArray** externalArrays,
                                 int numExt, void* stream);
#endif

  // CPU graph backend (oneDNN Graph or ACL Dynamic Fusion)
  GraphBackend* cpuGraphBackend_;
  bool cpuGraphBackendChecked_;

  Status executeSegmentWithCpuGraph(GraphSegment& seg, NDArray** externalArrays,
                                    int numExt, void* stream);
  GraphBackend* getCpuGraphBackend();
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_NATIVE_DYNAMIC_SHAPE_PLAN_H
