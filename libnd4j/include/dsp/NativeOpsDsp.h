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
// DSP (DynamicShapePlan) subsystem JNI bridge declarations.
//
// This header is parsed by JavaCPP to generate Java bindings (Nd4jCuda.java / Nd4jCpu.java).
// It is intentionally separate from NativeOps.h to avoid recompilation blast radius:
// NativeOps.h is included by most of the C++ codebase, so any change forces a full rebuild.
// This header is only included by the preset and the NativeOps_dsp.cu/.cpp implementation files.
//
// ALL new DSP/plan/graph/triton/NCCL JNI functions go here, NOT in NativeOps.h.
//

#ifndef NATIVEOPSDSP_H
#define NATIVEOPSDSP_H

#include <system/common.h>
#include <types/float16.h>
#include <graph/Context.h>

// Opaque type aliases used in JNI bridge declarations.
// These mirror the typedefs in NativeOps.h but are repeated here
// to avoid including NativeOps.h (widely included, causes rebuild blast).
// Duplicate typedefs to the same type are legal in C++11+.
#include <array/InteropDataBuffer.h>
typedef sd::graph::Context OpaqueContext;
typedef sd::NDArray* OpaqueNDArray;
typedef sd::NDArray** OpaqueNDArrayArr;
typedef sd::InteropDataBuffer OpaqueDataBuffer;

// ========================
// Native Graph Executor API
// ========================

/**
 * Compile a serialized DynamicShapePlan into a native C++ executor.
 * The serialized plan is a binary format produced by DynamicShapePlan.serialize() in Java.
 *
 * @param serializedPlan  Pointer to the serialized plan bytes
 * @param planSize  Size of the serialized plan in bytes
 * @return Opaque handle to the compiled plan, or nullptr on failure
 */
SD_LIB_EXPORT sd::Pointer compileDynamicShapePlan(sd::Pointer serializedPlan, sd::LongType planSize);

/**
 * Execute a compiled native plan.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param opContext  OpaqueContext with inputs set via setGraphContextInputArray(), outputs pre-allocated via setGraphContextOutputArray()
 * @param stream  CUDA stream pointer (nullptr for CPU)
 * @return 0 on success, non-zero on failure
 */
SD_LIB_EXPORT int executeDynamicShapePlan(
    sd::Pointer planHandle,
    OpaqueContext* opContext,
    sd::Pointer stream);

/**
 * Free a compiled native plan.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 */
SD_LIB_EXPORT void freeDynamicShapePlan(sd::Pointer planHandle);

/**
 * Create a new per-SameDiff native plan cache. Opaque handle owned by caller;
 * free via freeNativePlanCache().
 */
SD_LIB_EXPORT sd::Pointer createNativePlanCache();

/**
 * Destroy a cache created by createNativePlanCache(). Frees all entries.
 */
SD_LIB_EXPORT void freeNativePlanCache(sd::Pointer cacheHandle);

/**
 * Clear all entries from a cache (does NOT delete the cache itself).
 */
SD_LIB_EXPORT void clearNativePlanCacheHandle(sd::Pointer cacheHandle);

/**
 * Get-or-build a NativeDynamicShapePlan keyed by (outputSet, placeholder shape-info, mode).
 *
 * @param cacheHandle           cache from createNativePlanCache (non-null)
 * @param planBytes             serialized Java DynamicShapePlan (for cold-miss build)
 * @param planBytesLen          byte count of planBytes
 * @param outputNames           packed C-string array of output variable names, UTF-8, NUL-separated
 * @param numOutputs            number of output names
 * @param phShapeInfoPtrs       array of shape-info pointers (from ConstantShapeHelper); identity = key equality
 * @param numPlaceholders       length of phShapeInfoPtrs
 * @param graphExecutionMode    GraphExecutionMode ordinal — each mode gets its own plan
 * @return                      NativeDynamicShapePlan* as opaque sd::Pointer; owned by cache
 */
SD_LIB_EXPORT sd::Pointer dispatchNativePlan(sd::Pointer cacheHandle,
                                             sd::Pointer planBytes,
                                             sd::LongType planBytesLen,
                                             sd::Pointer outputNames,
                                             sd::LongType numOutputs,
                                             sd::Pointer phShapeInfoPtrs,
                                             sd::LongType numPlaceholders,
                                             int graphExecutionMode);

/**
 * Unpin a plan handle, making it eligible for LRU eviction.
 * Must be called when Java swaps to a different plan handle or closes
 * the executor. Paired with the automatic pinning done by dispatchNativePlan().
 *
 * @param cacheHandle  cache from createNativePlanCache (non-null)
 * @param planHandle   plan handle from dispatchNativePlan (safe to pass null — no-op)
 */
SD_LIB_EXPORT void unpinNativePlan(sd::Pointer cacheHandle, sd::Pointer planHandle);

/**
 * Clear shape caches in a compiled plan.
 * Must be called when a session resets to avoid stale GPU memory references.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 */
SD_LIB_EXPORT void clearDynamicShapePlanCaches(sd::Pointer planHandle);

/**
 * Force-clear ALL shape caches unconditionally (including static slots).
 * Use for session reset or model reload scenarios.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 */
SD_LIB_EXPORT void clearAllDynamicShapePlanCachesForce(sd::Pointer planHandle);

/**
 * Release all GPU memory held by intermediate computation results while keeping
 * the plan structure alive. Frees CUDA graph replay handles and associated
 * replay resources,
 * cuBLAS workspace, and non-weight output slot NDArrays. The plan enters a
 * "cold" state and will re-warm on the next execute() call.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return the number of intermediate NDArrays freed
 */
SD_LIB_EXPORT int releaseGpuIntermediates(sd::Pointer planHandle);

// ─── Replay diagnostics (Phase 2) ──────────────────────────────────────────

/**
 * Get the replay schedule signature hash for a segment.
 * Returns the FNV-1a hash encoding of the ordered replay unit list
 * (unit kinds, slot ranges, op types). Zero if the segment has no replay.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param segIdx     Segment index (0-based)
 * @return FNV-1a hash of replay schedule, or 0 if no replay
 */
SD_LIB_EXPORT unsigned long long getPlanReplaySignatureHash(sd::Pointer planHandle, int segIdx);

/**
 * Get the number of replay units for a segment after consolidation.
 * Returns the count of ordered replay units (Triton islands + prep units).
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param segIdx     Segment index (0-based)
 * @return Number of replay units, or 0 if no replay
 */
SD_LIB_EXPORT int getPlanReplayUnitCount(sd::Pointer planHandle, int segIdx);

/**
 * Get the current plan execution phase (0=SLOT_BY_SLOT, 1=SHAPES_FROZEN,
 * 2=REPLAYING).
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Current plan phase (0-2), or -1 on error
 */
SD_LIB_EXPORT int getPlanPhase(sd::Pointer planHandle);

/**
 * Get the execution count for a segment (number of times executed).
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param segIdx     Segment index (0-based)
 * @return Execution count, or -1 on error
 */
SD_LIB_EXPORT int getSegmentExecutionCount(sd::Pointer planHandle, int segIdx);

/**
 * Get the number of segments in the plan.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Number of segments, or -1 on error
 */
SD_LIB_EXPORT int getPlanSegmentCount(sd::Pointer planHandle);

/**
 * Query whether the plan's compilation has been sealed (first phaseCompile()
 * has completed). Returns 1 if sealed, 0 if not, -1 on error.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 */
SD_LIB_EXPORT int isPlanCompilationSealed(sd::Pointer planHandle);

/**
 * Returns the count of compileSegment() calls that happened AFTER compilation
 * was sealed. Any value > 0 is a correctness red flag — it means the plan
 * re-compiled a segment mid-execution which breaks the freeze/capture contract.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Mid-execution compile count, or -1 on error
 */
SD_LIB_EXPORT long long getPlanMidExecutionCompileCount(sd::Pointer planHandle);

/**
 * Resets the mid-execution compile counter to zero. Used by tests that want
 * to bracket a section and assert zero recompiles happened inside it.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 */
SD_LIB_EXPORT void resetPlanMidExecutionCompileCount(sd::Pointer planHandle);

// ═══════════════════════════════════════════════════════════════════════════════
// FrozenPlan Hierarchy API (Step 6) — New unified execution interface
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Execute a FrozenPlan. Unified entry point that auto-manages build → seal → replay.
 * Delegates to executeDynamicShapePlan internally (Step 1 compatibility).
 *
 * @param planHandle  Handle from dispatchNativePlan()
 * @param opContext   OpaqueContext with inputs/outputs set
 * @param stream      CUDA stream pointer (nullptr for CPU)
 * @return 0 on success, non-zero on failure
 */
SD_LIB_EXPORT int executeFrozenPlan(
    sd::Pointer planHandle,
    OpaqueContext* opContext,
    sd::Pointer stream);

/**
 * Check whether a FrozenPlan is sealed (all segments completed build phase).
 *
 * @param planHandle  Handle from dispatchNativePlan()
 * @return 1 if sealed, 0 if still building, -1 if invalid
 */
SD_LIB_EXPORT int isFrozenPlanSealed(sd::Pointer planHandle);

/**
 * Get the build pass count for a FrozenPlan.
 * 0 = needs warmup, 1 = needs compile/capture, 2+ = sealed (replay only).
 *
 * @param planHandle  Handle from dispatchNativePlan()
 * @return build pass count, -1 if invalid
 */
SD_LIB_EXPORT int getFrozenPlanBuildPassCount(sd::Pointer planHandle);

/**
 * Get the segment executor phase for a specific segment.
 * Returns: 0=BUILDING, 1=SEALED, 2=FAILED, -1=invalid.
 *
 * @param planHandle  Handle from dispatchNativePlan()
 * @param segIdx      Segment index (0-based)
 * @return Phase code
 */
SD_LIB_EXPORT int getSegmentExecutorPhase(sd::Pointer planHandle, int segIdx);

// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Load a model from an SDZ (ZIP) or SDNB file entirely in C++.
 *
 * @param filePath  Path to the .sdz or .sdnb file
 * @return Opaque handle to the loaded model, or nullptr on failure
 */
SD_LIB_EXPORT sd::Pointer loadModelFromFile(const char* filePath);

/**
 * Compile a loaded model into a native execution plan.
 *
 * @param modelHandle  Handle from loadModelFromFile()
 * @param requestedOutputNames  Array of output variable name strings
 * @param numOutputs  Number of requested outputs
 * @return Opaque plan handle, or nullptr on failure
 */
SD_LIB_EXPORT sd::Pointer compileModelPlan(
    sd::Pointer modelHandle,
    sd::Pointer requestedOutputNames, int numOutputs);

/**
 * Free a loaded model.
 *
 * @param modelHandle  Handle from loadModelFromFile()
 */
SD_LIB_EXPORT void freeLoadedModel(sd::Pointer modelHandle);

/**
 * Get the number of external inputs required by a compiled plan.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Number of external inputs
 */
SD_LIB_EXPORT int getPlanNumExternalInputs(sd::Pointer planHandle);

/**
 * Get the number of requested outputs in a compiled plan.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Number of requested outputs
 */
SD_LIB_EXPORT int getPlanNumRequestedOutputs(sd::Pointer planHandle);

/**
 * Get the number of slots (ops) in a compiled plan.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Number of slots
 */
SD_LIB_EXPORT int getPlanNumSlots(sd::Pointer planHandle);

/**
 * Enable or disable CUDA Graphs for a compiled plan.
 * When enabled, capturable segments are captured as CUDA graphs on the
 * second execution and replayed on subsequent calls with matching shapes.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param enabled  true to enable, false to disable
 */
SD_LIB_EXPORT void setPlanCudaGraphsEnabled(sd::Pointer planHandle, bool enabled);

/**
 * Set the minimum segment size for CUDA graph capture.
 * Segments smaller than this are always executed slot-by-slot.
 * Default: 10. Set to 1 for testing.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param minSize  Minimum number of slots for capture (clamped to >=1)
 */
SD_LIB_EXPORT void setPlanMinCaptureSegmentSize(sd::Pointer planHandle, int minSize);

/**
 * Set maximum segment size for CUDA graph capture. Large capturable segments
 * are split into sub-segments of at most this size to prevent OOM during capture.
 * Default: 300.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param maxSize     Maximum slots per capture segment (0 for unlimited)
 */
SD_LIB_EXPORT void setPlanMaxCaptureSegmentSize(sd::Pointer planHandle, int maxSize);

/**
 * Enable/disable "shapes frozen" mode for a compiled plan.
 * When frozen, shape inference and cache clearing are skipped between executions.
 * Use during static KV decode where external input shapes are guaranteed constant.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param frozen      true to enable, false to disable
 */
SD_LIB_EXPORT void setPlanShapesFrozen(sd::Pointer planHandle, bool frozen);

/**
 * Enable/disable shape-only dry-run mode for a compiled plan.
 * When enabled, executeSlot() runs all dispatch infrastructure (shape caching,
 * frozen detection, output allocation, segment dispatch) but SKIPS op->execute().
 * Use to measure pure dispatch/infrastructure overhead separately from compute.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param enabled     true to enable shape-only mode, false to disable
 */
SD_LIB_EXPORT void setPlanShapeOnlyMode(sd::Pointer planHandle, bool enabled);

/**
 * Enable/disable execution timing breakdown logging for a compiled plan.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param enabled     true to enable timing, false to disable
 */
SD_LIB_EXPORT void setPlanExecutionTimingEnabled(sd::Pointer planHandle, bool enabled);

/**
 * Set the JIT compilation mode for DSP segment execution.
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param mode  0 = GRAPH_ONLY (default), 1 = JIT_ONLY, 2 = GRAPH_PLUS_JIT
 */
SD_LIB_EXPORT void setPlanJitMode(sd::Pointer planHandle, int mode);

/**
 * Set the graph execution mode for DSP execution.
 * Controls which backend is used for segment execution.
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param mode  0=AUTO, 1=SLOT_BY_SLOT, 2=CUDA_GRAPHS, 3=NVRTC_JIT, 4=PTX_JIT, 5=TRITON,
 *              6=MLX, 7=ARM_HYBRID, 8=NNAPI
 */
SD_LIB_EXPORT void setPlanGraphExecutionMode(sd::Pointer planHandle, int mode);

/**
 * Enable/disable trace logging for DSP execution decisions.
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param enabled     true to enable trace, false to disable
 */
SD_LIB_EXPORT void setPlanTraceEnabled(sd::Pointer planHandle, bool enabled);

/**
 * Set maximum sizes for specific output slots (KV cache pre-allocation).
 * When set, these slots will be pre-allocated at the specified maximum size,
 * keeping buffer addresses stable across all subsequent steps.
 * This enables CUDA graph capture for models with growing KV caches.
 *
 * @param planHandle       Handle from compileDynamicShapePlan()
 * @param numSlots         Number of slot entries
 * @param slotIndices      Array of output slot indices to pre-allocate
 * @param maxSizes         Array of maximum sizes (in number of elements, not bytes)
 */
SD_LIB_EXPORT void setPlanOutputSlotMaxSizes(sd::Pointer planHandle, sd::LongType numSlots,
                                               const int* slotIndices, const sd::LongType* maxSizes);

/**
 * Configure plan-managed KV scatter for CUDA-graph-compatible decode loops.
 *
 * After this call the plan executes a batched KV scatter after each execute(),
 * updating static KV buffers at the current position and incrementing the
 * position scalar. This eliminates the Java-side scatterNewEntries() round-trip.
 *
 * @param planHandle          The native plan handle
 * @param presentSlotIndices  Array of output slot indices for present KV tensors
 * @param staticKvBufferPtrs  Array of sd::Pointer (NDArray*) for static KV buffers
 * @param numPairs            Number of (present, static) pairs
 * @param dtypeInt            DataType code of the KV tensors
 * @param heads               Number of attention heads
 * @param srcSeqLen           Present sequence length (typically 1 for decode)
 * @param dstSeqLen           Static buffer sequence length (= maxKvLen)
 * @param dim                 Head dimension
 * @param kvPositionPtr       Pointer to device-accessible int64 position scalar
 */
SD_LIB_EXPORT void configurePlanKvScatter(sd::Pointer planHandle,
                                           const int* presentSlotIndices,
                                           const sd::Pointer* staticKvBufferPtrs,
                                           sd::LongType numPairs,
                                           int dtypeInt,
                                           sd::LongType heads,
                                           sd::LongType srcSeqLen,
                                           sd::LongType dstSeqLen,
                                           sd::LongType dim,
                                           sd::LongType* kvPositionPtr);

/**
 * Reset the KV cache position managed by the plan (e.g., after prefill).
 */
SD_LIB_EXPORT void resetPlanKvCachePosition(sd::Pointer planHandle, sd::LongType position);

/**
 * Get the current KV cache position managed by the plan.
 * Returns -1 if KV scatter is not configured.
 */
SD_LIB_EXPORT sd::LongType getPlanKvCachePosition(sd::Pointer planHandle);

/**
 * Get the number of graph segments in a compiled plan.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Number of segments, or -1 if handle is null
 */
SD_LIB_EXPORT int getPlanNumSegments(sd::Pointer planHandle);

/**
 * Get the number of segments that have been captured as CUDA graphs.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Number of captured graph segments
 */
SD_LIB_EXPORT int getPlanNumCapturedGraphSegments(sd::Pointer planHandle);

/**
 * Get the total number of CUDA graph replays across all segments.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Total graph replay count
 */
SD_LIB_EXPORT int getPlanTotalGraphReplays(sd::Pointer planHandle);

/**
 * Validate that the captured CUDA graph covers all ops in the plan.
 * Returns true if every op contributed at least one CUDA graph node.
 * Must be called after execution with debug/verbose mode active.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return true if all ops are captured, false if any are host-only
 */
SD_LIB_EXPORT bool validatePlanCapturedGraph(sd::Pointer planHandle);

/**
 * Get the count of host-only ops from the last capture audit.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Number of ops that contributed 0 CUDA graph nodes
 */
SD_LIB_EXPORT int getPlanNumHostOnlyOps(sd::Pointer planHandle);

/**
 * Get pipe-delimited names of host-only ops from the last capture audit.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Pipe-delimited string of op names (thread-local static storage)
 */
SD_LIB_EXPORT const char* getPlanHostOnlyOpNames(sd::Pointer planHandle);

/**
 * Print the full CUDA graph contents and capture audit to stderr.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 */
SD_LIB_EXPORT void printPlanCapturedGraphDebug(sd::Pointer planHandle);

/**
 * Get detailed capture statistics as a formatted string.
 * Returns: "captured=N|oomRetrying=N|permFailed=N|nonCapt=N|tooSmall=N|addrUnstable=N"
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Thread-local static buffer with stats string
 */
SD_LIB_EXPORT const char* getPlanCaptureStats(sd::Pointer planHandle);

// =============================================================================
// Per-Segment Replay State
// =============================================================================

/**
 * Get replay state for a specific segment.
 * @return ReplayState as int: 0=EMPTY, 1=CAPTURING, 2=CAPTURED, 3=READY, 4=ERROR, -1=no handle
 */
SD_LIB_EXPORT int getPlanSegmentReplayState(sd::Pointer planHandle, int segmentIdx);

/**
 * Get replay count for a specific segment (number of graph replays).
 */
SD_LIB_EXPORT int getPlanSegmentReplayCount(sd::Pointer planHandle, int segmentIdx);

/**
 * Get backend name for a specific segment ("CUDA", "CPU", or "").
 */
SD_LIB_EXPORT const char* getPlanSegmentBackendName(sd::Pointer planHandle, int segmentIdx);

/**
 * Get statistics JSON for a specific segment.
 * Returns: {"numOperations":N,"replayCount":N,"backendName":"..."}
 */
SD_LIB_EXPORT const char* getPlanSegmentStatisticsJson(sd::Pointer planHandle, int segmentIdx);

/**
 * Get total execution count for a specific segment.
 */
SD_LIB_EXPORT int getPlanSegmentExecutionCount(sd::Pointer planHandle, int segmentIdx);

/**
 * Check if segment is eligible for capture.
 */
SD_LIB_EXPORT bool isPlanSegmentCapturable(sd::Pointer planHandle, int segmentIdx);

/**
 * Check if capture permanently failed for a segment.
 */
SD_LIB_EXPORT bool isPlanSegmentCaptureFailed(sd::Pointer planHandle, int segmentIdx);

/**
 * Get execution phase for a specific segment.
 * Returns ExecutionPhase as int: 0=WARMUP, 1=COMPILING, 2=COMPILED, 3=REPLAYING, 4=SLOT_BY_SLOT, -1=invalid
 */
SD_LIB_EXPORT int getPlanSegmentExecutionPhase(sd::Pointer planHandle, int segmentIdx);

// =============================================================================
// Plan-Level Phase Tracking
// =============================================================================

/**
 * Check if all buffer pointers are stable (same addresses across executions).
 * Returns 1 if stable, 0 if not, -1 if invalid handle.
 */
SD_LIB_EXPORT int getPlanPointersStable(sd::Pointer planHandle);

/**
 * Get the number of executions since shapes were frozen.
 * Returns -1 if shapes are not frozen.
 */
SD_LIB_EXPORT int getPlanFrozenExecutionCount(sd::Pointer planHandle);

/**
 * Get the slot state for a specific slot.
 * Returns SlotState as int: 0=WARMUP, 1=SHAPE_CACHED,
 *   2=FROZEN, 3=FROZEN_CONSTANT. Returns -1 if invalid.
 */
SD_LIB_EXPORT int getPlanSlotState(sd::Pointer planHandle, int slotIdx);

// =============================================================================
// Per-Slot Op Detail (for DSP Debug Framework)
// =============================================================================

/**
 * Get the op name for a specific slot. Returns "" if invalid.
 * Caller must NOT free the returned string (static lifetime).
 */
SD_LIB_EXPORT const char* getPlanSlotOpName(sd::Pointer planHandle, int slotIdx);

/**
 * Get per-slot flags as a bitmask:
 *   bit 0: isViewCapableOp
 *   bit 1: isDataDependent
 *   bit 2: outputShapeDependsOnInputValues
 *   bit 3: isIdentityOp
 *   bit 4: inPlaceFused
 *   bit 5: isFusedChainHead
 *   bit 6: isFusedChainTail
 *   bit 7: needsZeroedOutput
 *   bit 8: needsIntLongSync
 *   bit 9: shapeStatic
 *   bit 10: frozenConstantSlot (state >= FROZEN_CONSTANT)
 * Returns -1 if invalid.
 */
SD_LIB_EXPORT int getPlanSlotFlags(sd::Pointer planHandle, int slotIdx);

/**
 * Get input/output counts for a slot. Returns via out params.
 * Returns 0 on success, -1 if invalid.
 */
SD_LIB_EXPORT int getPlanSlotIOCounts(sd::Pointer planHandle, int slotIdx,
                                       int* numInputsOut, int* numOutputsOut);

// =============================================================================
// External Input Variable Management
// =============================================================================

/**
 * Mark an external input as variable (participates in D2D staging).
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param extIdx      External input index
 */
SD_LIB_EXPORT void markPlanExternalInputVariable(sd::Pointer planHandle, int extIdx);

/**
 * Get number of cached variable ext input indices (fast-path list).
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Number of cached indices, or -1 if invalid handle
 */
SD_LIB_EXPORT int getPlanNumCachedVariableExtIndices(sd::Pointer planHandle);

/**
 * Get the i-th cached variable ext input index.
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param i           Index into the cached list
 * @return The ext input index, or -1 if out of range or invalid handle
 */
SD_LIB_EXPORT int getPlanCachedVariableExtIndex(sd::Pointer planHandle, int i);

/**
 * Mark an external input as a placeholder (implies variable + triggers H2D sync).
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param extIdx      External input index
 */
SD_LIB_EXPORT void markPlanExternalInputPlaceholder(sd::Pointer planHandle, int extIdx);

/**
 * Check if ext[extIdx] is classified as variable (participates in staging D2D).
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param extIdx      External input index
 * @return true if variable, false otherwise
 */
SD_LIB_EXPORT bool getPlanIsExternalInputVariable(sd::Pointer planHandle, int extIdx);

/**
 * Check if ext[extIdx] is classified as placeholder (forces H2D sync).
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param extIdx      External input index
 * @return true if placeholder, false otherwise
 */
SD_LIB_EXPORT bool getPlanIsExternalInputPlaceholder(sd::Pointer planHandle, int extIdx);

/**
 * Get the count of external inputs currently classified as variable.
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Count, or 0 if invalid handle
 */
SD_LIB_EXPORT int getPlanNumVariableExternalInputs(sd::Pointer planHandle);

/**
 * Get the plan's total execution count.
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return Execution count, or -1 if invalid handle
 */
SD_LIB_EXPORT int getPlanExecuteCount(sd::Pointer planHandle);

// =============================================================================
// Per-Segment Pointer Tracking
// =============================================================================

/**
 * Get tracked external input pointer addresses for a segment as JSON.
 * Returns: [{"inputIdx":0,"capturedAddr":"0x...","currentAddr":"0x...","match":true}, ...]
 */
SD_LIB_EXPORT const char* getPlanSegmentTrackedPointers(sd::Pointer planHandle, int segmentIdx);

/**
 * Get number of capture buffers for a segment.
 */
SD_LIB_EXPORT int getPlanSegmentNumCaptureBuffers(sd::Pointer planHandle, int segmentIdx);

/**
 * Get capture buffer descriptors as JSON for a segment.
 */
SD_LIB_EXPORT const char* getPlanSegmentCaptureBuffersJson(sd::Pointer planHandle, int segmentIdx);

/**
 * Get number of pinned host pointers held by segment's replay handle.
 */
SD_LIB_EXPORT int getPlanSegmentNumHostPointers(sd::Pointer planHandle, int segmentIdx);

// =============================================================================
// Replay Cache Management
// =============================================================================

/**
 * Check if replay cache is enabled.
 */
SD_LIB_EXPORT bool isReplayCacheEnabled();

/**
 * Get number of replay cache hits.
 */
SD_LIB_EXPORT int getReplayCacheHits();

/**
 * Get number of replay cache misses.
 */
SD_LIB_EXPORT int getReplayCacheMisses();

/**
 * Clear all replay cache entries.
 */
SD_LIB_EXPORT void clearReplayCache();

/**
 * Get replay cache directory path.
 */
SD_LIB_EXPORT const char* getReplayCacheDir();

/**
 * Get per-device replay cache statistics as JSON.
 */
SD_LIB_EXPORT const char* getReplayCacheDeviceStatsJson();

/**
 * Get replay cache entry count for a specific device.
 */
SD_LIB_EXPORT int getReplayCacheDeviceEntryCount(int deviceType, int deviceIndex);

/**
 * Clear replay cache for a specific device.
 */
SD_LIB_EXPORT void clearReplayCacheForDevice(int deviceType, int deviceIndex);

/**
 * Migrate replay cache between compatible devices.
 */
SD_LIB_EXPORT bool migrateReplayCache(int fromType, int fromIdx, int toType, int toIdx);

/**
 * Prune stale device cache entries.
 */
SD_LIB_EXPORT int pruneStaleReplayCacheDevices();

/**
 * Load replay cache for a specific device into a plan.
 */
SD_LIB_EXPORT int loadReplayCacheForDevice(sd::Pointer planHandle, int deviceType, int deviceIndex);

/**
 * Get all cached device keys as JSON.
 */
SD_LIB_EXPORT const char* getReplayCachedDevicesJson();

// =============================================================================
// Backend Plan Management
// =============================================================================

/**
 * Get available backends as JSON array.
 */
SD_LIB_EXPORT const char* getPlanAvailableBackends(sd::Pointer planHandle);

/**
 * Get which backend compiled a specific segment.
 */
SD_LIB_EXPORT const char* getPlanSegmentCompiledBackend(sd::Pointer planHandle, int segIdx);

/**
 * Get compilation audit JSON for a segment.
 */
SD_LIB_EXPORT const char* getPlanSegmentCompilationAudit(sd::Pointer planHandle, int segIdx);

/**
 * Invalidate compiled cache for a specific segment.
 */
SD_LIB_EXPORT void invalidatePlanSegmentCache(sd::Pointer planHandle, int segIdx);

/**
 * Invalidate all caches for a specific backend type.
 */
SD_LIB_EXPORT void invalidatePlanBackendCaches(sd::Pointer planHandle, const char* backendName);

/**
 * Get aggregated cache stats JSON for all backends.
 */
SD_LIB_EXPORT const char* getPlanBackendCacheStats(sd::Pointer planHandle);

/**
 * Override backend selection for a segment.
 */
SD_LIB_EXPORT void setPlanSegmentBackendOverride(sd::Pointer planHandle, int segIdx, const char* backendName);

/**
 * Set backend priority order (comma-separated names).
 */
SD_LIB_EXPORT void setPlanBackendPriority(sd::Pointer planHandle, const char* priorityList);

/**
 * Export the CUDA graph visualization to Chrome trace format.
 * The output JSON file can be loaded in chrome://tracing for detailed timeline analysis.
 * Similar to PyTorch's torch.cuda.CUDAGraph.debug_dump() functionality.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param outputPath  Output file path (should end with .json)
 * @return true on success
 */
SD_LIB_EXPORT bool exportPlanCudaGraphChromeTrace(sd::Pointer planHandle, const char* outputPath);

/**
 * Export the CUDA graph visualization to HTML format.
 * Creates a standalone HTML file with interactive visualization.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param outputPath  Output HTML file path
 * @return true on success
 */
SD_LIB_EXPORT bool exportPlanCudaGraphHtml(sd::Pointer planHandle, const char* outputPath);

/**
 * Dump all CUDA graph debug files (DOT, JSON, HTML, nodes JSON).
 * Creates: {outputPath}.dot, {outputPath}.json, {outputPath}.html, {outputPath}_nodes.json
 * PyTorch-style debug dump similar to torch.cuda.CUDAGraph.debug_dump().
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param outputPath  Base path for output files (without extension)
 * @return true on success
 */
SD_LIB_EXPORT bool debugDumpPlanCudaGraph(sd::Pointer planHandle, const char* outputPath);

/**
 * Get the CUDA graph execution timeline as a JSON string in Chrome trace format.
 * For programmatic access to the timeline data.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @return JSON string, or empty string if no graph data
 */
SD_LIB_EXPORT const char* getPlanCudaGraphChromeTraceJson(sd::Pointer planHandle);

/**
 * Clear the CUDA graph execution timeline history.
 * Useful to reset timing data between profiling sessions.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 */
SD_LIB_EXPORT void clearPlanCudaGraphTimeline(sd::Pointer planHandle);

// =============================================================================
// Triton GPU Backend Counters
// =============================================================================

/**
 * Returns true if Triton backend support is compiled and available at runtime.
 */
SD_LIB_EXPORT bool isTritonAvailable();

/**
 * Get the total number of Triton kernel launches since the backend was initialized.
 * Returns 0 if Triton is not available.
 */
SD_LIB_EXPORT sd::LongType getTritonKernelLaunchCount();

/**
 * Get the total number of Triton PTX cache hits since the backend was initialized.
 * Returns 0 if Triton is not available.
 */
SD_LIB_EXPORT sd::LongType getTritonCacheHitCount();

/**
 * Reset all Triton execution counters to zero.
 */
SD_LIB_EXPORT void resetTritonCounters();

/**
 * Invalidate all cached Triton compiled kernels (frees CUmodule GPU memory).
 */
SD_LIB_EXPORT void invalidateTritonCache();

/**
 * Export the Triton kernel disk cache to a shareable .tkcache bundle (STORED ZIP).
 * Returns the number of kernels exported, or negative on error.
 */
SD_LIB_EXPORT int exportTritonCacheBundle(const char* outputPath);

/**
 * Import a .tkcache bundle into the Triton override directory.
 * Kernels from bundles take priority over on-disk cache.
 * @param validateArch  If true, reject bundles compiled for incompatible GPU architectures.
 * @return Number of kernels imported, -1 on error, -2 on architecture mismatch.
 */
SD_LIB_EXPORT int importTritonCacheBundle(const char* bundlePath, bool validateArch);

/**
 * Read and return the manifest JSON from a .tkcache bundle without importing.
 * Returns a JSON string (thread-local buffer, valid until next call).
 */
SD_LIB_EXPORT const char* inspectTritonCacheBundle(const char* bundlePath);

// =============================================================================
// DSP Diagnostics
// =============================================================================

/**
 * Set DSP diagnostic categories (replaces existing mask).
 * Categories are bitfield values from DspDiagCategory enum.
 */
SD_LIB_EXPORT void dspDiagSetCategories(int mask);

/**
 * Enable additional DSP diagnostic categories (OR into existing mask).
 */
SD_LIB_EXPORT void dspDiagEnableCategories(int mask);

/**
 * Disable specific DSP diagnostic categories (AND-NOT from existing mask).
 */
SD_LIB_EXPORT void dspDiagDisableCategories(int mask);

/**
 * Get the currently enabled DSP diagnostic category bitmask.
 */
SD_LIB_EXPORT int dspDiagGetEnabledMask();

/**
 * Set DSP diagnostic output level.
 * @param level  0=SUMMARY (stats only), 1=DETAILED (per-step), 2=FULL (echo all to stdout)
 */
SD_LIB_EXPORT void dspDiagSetLevel(int level);

/**
 * Set DSP diagnostic JSON output file path.
 */
SD_LIB_EXPORT void dspDiagSetJsonPath(const char* path);

/**
 * Record a diagnostic event from Java.
 */
SD_LIB_EXPORT void dspDiagRecordJavaEvent(int category, int slotId, int segmentId,
                                            const char* opName, sd::LongType timingUs,
                                            const char* message);

/**
 * Get human-readable plan execution report.
 */
SD_LIB_EXPORT const char* dspDiagGetPlanReport();

/**
 * Get JSON-formatted diagnostic report.
 */
SD_LIB_EXPORT const char* dspDiagGetJsonReport();

/**
 * Clear all diagnostic state (ring buffer, stats, snapshots).
 */
SD_LIB_EXPORT void dspDiagClear();

/**
 * Get total diagnostic step count.
 */
SD_LIB_EXPORT int dspDiagGetStepCount();

/**
 * Get total event count across all categories.
 */
SD_LIB_EXPORT long long dspDiagGetTotalEventCount();

/**
 * Get event count for a specific category by index.
 */
SD_LIB_EXPORT long long dspDiagGetCategoryEventCount(int categoryIndex);

/**
 * Validate plan outputs after execution. Checks each output for NaN, Inf, all-zeros, or null.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param flagsOut    int array of size numRequestedOutputs, filled with DSP_VALIDATE_* bitmask per output
 * @return number of outputs with validation issues
 */
SD_LIB_EXPORT int dspValidateOutputs(sd::Pointer planHandle, int* flagsOut);

/**
 * Detect stale outputs by comparing current outputs against previous step norms.
 *
 * @param planHandle  Handle from compileDynamicShapePlan()
 * @param prevNorms   float array of size numRequestedOutputs (updated in-place with current norms)
 * @param staleOut    bool array of size numRequestedOutputs, set to true for stale outputs
 * @param epsilon     threshold below which norm difference is considered stale
 * @return number of stale outputs detected
 */
SD_LIB_EXPORT int dspDetectStaleOutputs(sd::Pointer planHandle, float* prevNorms, bool* staleOut, float epsilon);

// =============================================================================
// DSP Freeze Config
// =============================================================================

SD_LIB_EXPORT void setDspFreezeMergeSegments(bool enable);
SD_LIB_EXPORT void setDspFreezeRecompile(bool enable);
SD_LIB_EXPORT bool getDspFreezeMergeSegments();
SD_LIB_EXPORT bool getDspFreezeRecompile();

// =============================================================================
// Per-Segment Summary
// =============================================================================

/**
 * Get segments summary as JSON.
 */
SD_LIB_EXPORT const char* getPlanSegmentsSummaryJson(sd::Pointer planHandle);

// =============================================================================
// Staging Buffer Introspection
// =============================================================================

/**
 * Number of staging buffers allocated for variable ext inputs.
 */
SD_LIB_EXPORT int getPlanNumStagingBuffers(sd::Pointer planHandle);

/**
 * Device address of the staging buffer for ext[extIdx], or 0.
 */
SD_LIB_EXPORT long long getPlanStagingBufferAddress(sd::Pointer planHandle, int extIdx);

/**
 * Device address the CUDA graph reads from for ext[extIdx].
 */
SD_LIB_EXPORT long long getPlanEffectiveExternalAddress(sd::Pointer planHandle, int extIdx);

/**
 * Device address of the last externalArrays[extIdx] passed to execute/executeSteadyState.
 */
SD_LIB_EXPORT long long getPlanLastExternalInputAddress(sd::Pointer planHandle, int extIdx);

/**
 * Get the staging buffer as an OpaqueNDArray for ext[extIdx].
 * Returns null if no staging buffer exists.
 */
SD_LIB_EXPORT OpaqueNDArray getPlanStagingBufferArray(sd::Pointer planHandle, int extIdx);

/**
 * Atomically copy staging buffer content for ext[extIdx] into dstBuffer.
 * This avoids the stale-pointer race of extracting specialBuffer() then copying separately.
 * Returns: 0 = success, -1 = no plan, -2 = no staging buffer, -3 = copy failed.
 */
SD_LIB_EXPORT int copyPlanStagingToBuffer(sd::Pointer planHandle, int extIdx, OpaqueDataBuffer* dstBuffer);

// =============================================================================
// Slot Output Introspection
// =============================================================================

/**
 * Get a slot's output array as OpaqueNDArray.
 */
SD_LIB_EXPORT OpaqueNDArray getPlanSlotOutputArray(sd::Pointer planHandle, int slotIdx);

/**
 * Get the total number of output slots.
 */
SD_LIB_EXPORT int getTotalPlanOutputSlots(sd::Pointer planHandle);

/**
 * Get the monotonic write-generation counter for a slot.
 */
SD_LIB_EXPORT int getPlanSlotGeneration(sd::Pointer planHandle, int slotIdx);

// =============================================================================
// Replay Mode & Arg Generation
// =============================================================================

/**
 * Replay mode: 0=NONE, 1=MONOLITHIC, 2=COMPOSITE.
 */
SD_LIB_EXPORT int getPlanSegmentReplayMode(sd::Pointer planHandle, int segIdx);

/**
 * Current arg generation counter for a segment.
 */
SD_LIB_EXPORT long long getPlanSegmentArgGeneration(sd::Pointer planHandle, int segIdx);

/**
 * Arg generation captured when CUDA graph was recorded.
 */
SD_LIB_EXPORT long long getPlanSegmentCapturedArgGeneration(sd::Pointer planHandle, int segIdx);

/**
 * Whether a segment needs arg table refresh before replay.
 */
SD_LIB_EXPORT int getPlanSegmentNeedsArgRefresh(sd::Pointer planHandle, int segIdx);

/**
 * Hash key of captured input addresses for a segment.
 */
SD_LIB_EXPORT long long getPlanSegmentCapturedInputAddrKey(sd::Pointer planHandle, int segIdx);

// =============================================================================
// Per-Execution Stats
// =============================================================================

SD_LIB_EXPORT int getLastExecSegmentsWarmup(sd::Pointer planHandle);
SD_LIB_EXPORT int getLastExecSegmentsCaptured(sd::Pointer planHandle);
SD_LIB_EXPORT int getLastExecSegmentsReplayed(sd::Pointer planHandle);
SD_LIB_EXPORT int getLastExecSegmentsSlotBySlot(sd::Pointer planHandle);
SD_LIB_EXPORT int getLastExecSegmentsFailed(sd::Pointer planHandle);
SD_LIB_EXPORT int getLastExecSegmentsTotal(sd::Pointer planHandle);
SD_LIB_EXPORT int getLastExecSyncLevel(sd::Pointer planHandle);
SD_LIB_EXPORT int getLastExecStreamSyncCount(sd::Pointer planHandle);
SD_LIB_EXPORT int getLastExecConsecutiveUnchangedCount(sd::Pointer planHandle);

// =============================================================================
// Cross-Stream Testing API
// =============================================================================

SD_LIB_EXPORT sd::Pointer dspCreateTestStream();
SD_LIB_EXPORT void dspDestroyTestStream(sd::Pointer streamPtr);
SD_LIB_EXPORT int dspWriteDeviceBufferOnDefaultStream(sd::Pointer planHandle, int extIdx, sd::Pointer srcHost, long long numBytes);
SD_LIB_EXPORT int dspWriteDeviceBufferOnExplicitStream(sd::Pointer planHandle, int extIdx, sd::Pointer srcHost, long long numBytes, sd::Pointer streamPtr);
SD_LIB_EXPORT int dspSyncStream(sd::Pointer streamPtr);
SD_LIB_EXPORT int dspIsExtInputDeviceAuthoritative(sd::Pointer planHandle, int extIdx);
SD_LIB_EXPORT sd::Pointer dspGetExecutionStream(sd::Pointer planHandle);
SD_LIB_EXPORT sd::Pointer dspGetDefaultStream();

// =============================================================================
// NCCL Collective Communication Operations
// =============================================================================

/**
 * Initialize an NCCL communicator for a group of GPUs.
 *
 * @param numRanks   Number of ranks (GPUs) in the communicator
 * @param rankId     This process's rank (0-indexed)
 * @param deviceId   CUDA device ID for this rank
 * @return Opaque pointer to the NCCL communicator, or nullptr on failure
 */
SD_LIB_EXPORT sd::Pointer ncclCommInit(int numRanks, int rankId, int deviceId);

/**
 * Initialize an NCCL communicator from a unique ID (for multi-process).
 *
 * @param numRanks   Number of ranks
 * @param rankId     This process's rank
 * @param uniqueId   Pointer to an ncclUniqueId (128 bytes)
 * @return Opaque pointer to the NCCL communicator
 */
SD_LIB_EXPORT sd::Pointer ncclCommInitWithId(int numRanks, int rankId, sd::Pointer uniqueId);

/**
 * Generate a unique NCCL ID for multi-process initialization.
 *
 * @return Pointer to a newly allocated ncclUniqueId (caller must free)
 */
SD_LIB_EXPORT sd::Pointer ncclGetUniqueId();

/**
 * Destroy an NCCL communicator and release resources.
 *
 * @param commHandle Handle from ncclCommInit
 */
SD_LIB_EXPORT void ncclCommDestroy(sd::Pointer commHandle);

/**
 * AllReduce: sum all tensors across ranks, result available on all ranks.
 *
 * @param commHandle   NCCL communicator handle
 * @param sendBuf      Send buffer (OpaqueDataBuffer*)
 * @param recvBuf      Receive buffer (OpaqueDataBuffer*), can be same as sendBuf for in-place
 * @param numElements  Number of elements
 * @param dataType     Data type (sd::DataType ordinal)
 * @param reduceOp     Reduction operation (0=SUM, 1=PROD, 2=MAX, 3=MIN, 4=AVG)
 * @param stream       CUDA stream pointer (nullptr for default stream)
 * @return 0 on success, non-zero on failure
 */
SD_LIB_EXPORT int ncclDoAllReduce(sd::Pointer commHandle,
                                   sd::Pointer sendBuf, sd::Pointer recvBuf,
                                   sd::LongType numElements, int dataType,
                                   int reduceOp, sd::Pointer stream);

/**
 * AllGather: gather data from all ranks, result available on all ranks.
 *
 * @param commHandle   NCCL communicator handle
 * @param sendBuf      Send buffer (this rank's data)
 * @param recvBuf      Receive buffer (must hold numRanks * sendCount elements)
 * @param sendCount    Number of elements per rank
 * @param dataType     Data type ordinal
 * @param stream       CUDA stream pointer
 * @return 0 on success, non-zero on failure
 */
SD_LIB_EXPORT int ncclDoAllGather(sd::Pointer commHandle,
                                   sd::Pointer sendBuf, sd::Pointer recvBuf,
                                   sd::LongType sendCount, int dataType,
                                   sd::Pointer stream);

/**
 * ReduceScatter: reduce then scatter result across ranks.
 *
 * @param commHandle   NCCL communicator handle
 * @param sendBuf      Send buffer (full data)
 * @param recvBuf      Receive buffer (this rank's shard)
 * @param recvCount    Number of elements per rank after scatter
 * @param dataType     Data type ordinal
 * @param reduceOp     Reduction operation
 * @param stream       CUDA stream pointer
 * @return 0 on success, non-zero on failure
 */
SD_LIB_EXPORT int ncclDoReduceScatter(sd::Pointer commHandle,
                                       sd::Pointer sendBuf, sd::Pointer recvBuf,
                                       sd::LongType recvCount, int dataType,
                                       int reduceOp, sd::Pointer stream);

/**
 * Send data to a specific peer rank.
 *
 * @param commHandle   NCCL communicator handle
 * @param sendBuf      Send buffer
 * @param numElements  Number of elements to send
 * @param dataType     Data type ordinal
 * @param peerRank     Destination rank
 * @param stream       CUDA stream pointer
 * @return 0 on success, non-zero on failure
 */
SD_LIB_EXPORT int ncclDoSend(sd::Pointer commHandle,
                              sd::Pointer sendBuf, sd::LongType numElements,
                              int dataType, int peerRank, sd::Pointer stream);

/**
 * Receive data from a specific peer rank.
 *
 * @param commHandle   NCCL communicator handle
 * @param recvBuf      Receive buffer
 * @param numElements  Number of elements to receive
 * @param dataType     Data type ordinal
 * @param peerRank     Source rank
 * @param stream       CUDA stream pointer
 * @return 0 on success, non-zero on failure
 */
SD_LIB_EXPORT int ncclDoRecv(sd::Pointer commHandle,
                              sd::Pointer recvBuf, sd::LongType numElements,
                              int dataType, int peerRank, sd::Pointer stream);

/**
 * Begin a group of NCCL operations (for fusing multiple collectives).
 * @return 0 on success
 */
SD_LIB_EXPORT int ncclGroupStart();

/**
 * End a group of NCCL operations.
 * @return 0 on success
 */
SD_LIB_EXPORT int ncclGroupEnd();

// ========================
// Plan Cache Shutdown Guard
// ========================

/**
 * Mark the native plan cache subsystem as shutting down.
 * When set, unpinPlan() and evictIfOverBudgetLocked() skip plan deletion
 * entirely — the OS reclaims all memory on process exit.
 *
 * This prevents SIGSEGV from CUDA API calls (cudaStreamDestroy, cudaFree,
 * cudaGraphExecDestroy) racing with JVM shutdown tearing down the CUDA context.
 *
 * Must be called from the JVM shutdown hook BEFORE DeallocatorService shutdown.
 *
 * @param inProgress true to mark shutdown in progress, false otherwise
 */
SD_LIB_EXPORT void setPlanCacheShutdownInProgress(bool inProgress);

// ========================
// Buffer Coloring Introspection
// ========================

/** Whether buffer coloring is currently applied on this plan. */
SD_LIB_EXPORT bool getPlanBufferColoringApplied(sd::Pointer planHandle);

/** Number of colors assigned by coloring (0 if not computed). */
SD_LIB_EXPORT int getPlanBufferColoringNumColors(sd::Pointer planHandle);

/** Estimated bytes saved by coloring. */
SD_LIB_EXPORT sd::LongType getPlanBufferColoringBytesSaved(sd::Pointer planHandle);

/** Color assigned to a specific slot (-1 if uncolored). */
SD_LIB_EXPORT int getPlanSlotColor(sd::Pointer planHandle, int slotIdx);

// ========================
// Buffer Pool Introspection
// ========================

/** Total bytes currently pooled on the given device. */
SD_LIB_EXPORT sd::LongType getBufferPoolPooledBytes(int deviceId);

/** Number of buffers currently in the pool on the given device. */
SD_LIB_EXPORT int getBufferPoolPooledCount(int deviceId);

/** Lifetime acquire count on the given device. */
SD_LIB_EXPORT sd::LongType getBufferPoolTotalAcquired(int deviceId);

/** Lifetime reuse count (acquires satisfied from pool) on the given device. */
SD_LIB_EXPORT sd::LongType getBufferPoolTotalReused(int deviceId);

// ========================
// Sync-free Buffer Fingerprint Ring
// ========================

/**
 * Drain the fingerprint ring: D2H copy device→host (synchronous, call ONLY
 * after the decode loop, never during capture/replay).
 * No-op if BUF_FP_RING env not set or ring already drained.
 */
SD_LIB_EXPORT void drainPlanFingerprintRing(sd::Pointer planHandle);

/**
 * Return JSON string describing per-step XOR fingerprints of all tracked
 * buffers. Call after drainPlanFingerprintRing(). Returns "null" if not
 * enabled or not drained.
 */
SD_LIB_EXPORT const char* getPlanFingerprintJson(sd::Pointer planHandle);

#endif // NATIVEOPSDSP_H
