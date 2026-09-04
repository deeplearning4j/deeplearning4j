/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Vulkan NativeOps DSP portable/control-plane bridge.
 */
#include <dsp/NativeOpsDsp.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspPhaseUtils.h>
#include <graph/DspSegmentLifecycle.h>
#include <graph/DspVerifyUtils.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/NativePlanCache.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/SdnbReader.h>
#include <graph/SdzReader.h>
#include <system/Environment.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <map>
#include <unordered_map>
#include <vector>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

using namespace sd;
using namespace sd::graph;


namespace {
inline NativeDynamicShapePlan* planOf(sd::Pointer handle) {
  return reinterpret_cast<NativeDynamicShapePlan*>(handle);
}

inline void setPlanError(int code, const char* message) {
  auto* error = sd::LaunchContext::defaultContext()->errorReference();
  error->setErrorCode(code);
  error->setErrorMessage(message == nullptr ? "" : message);
}

inline bool validSegment(NativeDynamicShapePlan* plan, int index) {
  return plan != nullptr && index >= 0 &&
         index < static_cast<int>(plan->getSegments().size());
}

std::string jsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '\\': escaped += "\\\\"; break;
      case '"': escaped += "\\\""; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default: escaped += c; break;
    }
  }
  return escaped;
}

template <typename Fn>
void forEachCompositeReplayHandle(const GraphSegment& segment, Fn&& fn) {
  for (const auto& handle :
       segment.exec.compositeReplaySchedule.mergedReplayHandles) {
    if (handle != nullptr) fn(handle.get());
  }
  for (const auto& handle :
       segment.exec.compositeReplaySchedule.compositeReplayHandles) {
    if (handle != nullptr) fn(handle.get());
  }
}

bool segmentHasReadyCompositeReplay(const GraphSegment& segment) {
  bool ready = false;
  forEachCompositeReplayHandle(segment, [&](const GraphReplayHandle* handle) {
    ready = ready || handle->isReady();
  });
  return ready;
}

bool segmentHasReadyReplay(const GraphSegment& segment) {
  return (segment.exec.replayHandle != nullptr &&
          segment.exec.replayHandle->isReady()) ||
         segmentHasReadyCompositeReplay(segment);
}

bool segmentHasErroredReplay(const GraphSegment& segment) {
  if (segment.exec.replayHandle != nullptr &&
      segment.exec.replayHandle->getState() == ReplayState::ERRORED) {
    return true;
  }
  bool errored = false;
  forEachCompositeReplayHandle(segment, [&](const GraphReplayHandle* handle) {
    errored = errored || handle->getState() == ReplayState::ERRORED;
  });
  return errored;
}

ReplayStatistics segmentReplayStatistics(const GraphSegment& segment) {
  ReplayStatistics result;
  const bool useComposite = segmentHasReadyCompositeReplay(segment);
  if (!useComposite) {
    return segment.exec.replayHandle != nullptr
        ? segment.exec.replayHandle->getStatistics()
        : result;
  }

  forEachCompositeReplayHandle(segment, [&](const GraphReplayHandle* handle) {
    if (!handle->isReady()) return;
    const ReplayStatistics stats = handle->getStatistics();
    result.numOperations += stats.numOperations;
    result.numMemoryOps += stats.numMemoryOps;
    result.estimatedMemory += stats.estimatedMemory;
    result.captureTimeMs += stats.captureTimeMs;
    result.lastReplayTimeMs += stats.lastReplayTimeMs;
    result.replayCount += stats.replayCount;
    if (result.deviceName.empty()) result.deviceName = stats.deviceName;
    if (result.apiVersion == 0) result.apiVersion = stats.apiVersion;
    result.memoryBudgetBytes += stats.memoryBudgetBytes;
  });
  return result;
}

const GraphReplayHandle* segmentRepresentativeReplay(
    const GraphSegment& segment) {
  const GraphReplayHandle* representative = nullptr;
  forEachCompositeReplayHandle(segment, [&](const GraphReplayHandle* handle) {
    if (representative == nullptr && handle->isReady()) {
      representative = handle;
    }
  });
  if (representative != nullptr) return representative;
  if (segment.exec.replayHandle != nullptr) {
    return segment.exec.replayHandle.get();
  }
  forEachCompositeReplayHandle(segment, [&](const GraphReplayHandle* handle) {
    if (representative == nullptr) representative = handle;
  });
  return representative;
}

int segmentReplayState(const GraphSegment& segment) {
  if (segmentHasReadyCompositeReplay(segment)) {
    return static_cast<int>(ReplayState::READY);
  }
  const GraphReplayHandle* replay = segmentRepresentativeReplay(segment);
  return replay != nullptr ? static_cast<int>(replay->getState()) : -1;
}

std::string planTraceJson(NativeDynamicShapePlan* plan) {
  std::vector<DspTraceEvent> events;
  if (plan != nullptr && plan->getTrace() != nullptr) {
    plan->getTrace()->forEach([&](const DspTraceEvent& event) {
      events.push_back(event);
    });
  }

  std::ostringstream json;
  json << "{\n  \"traceEvents\": [";
  bool first = true;
  for (auto it = events.rbegin(); it != events.rend(); ++it) {
    if (!first) json << ',';
    first = false;
    json << "\n    {\"name\":\""
         << DspExecutionTrace::traceEventKindName(it->kind)
         << "\",\"cat\":\"Vulkan DSP\",\"ph\":\"i\",\"s\":\"t\","
         << "\"pid\":0,\"tid\":" << static_cast<int>(it->segIdx)
         << ",\"ts\":" << (it->timestampNs / 1000)
         << ",\"args\":{\"segment\":" << static_cast<int>(it->segIdx)
         << ",\"backend\":" << static_cast<int>(it->backendId)
         << ",\"startSlot\":" << it->slotIdx
         << ",\"endSlot\":" << it->slotRangeEnd
         << ",\"executionCount\":" << it->execCount
         << ",\"addressHash\":" << it->bufAddrHash
         << ",\"detail\":" << it->detail << "}}";
  }
  if (!events.empty()) json << '\n';
  json << "  ]\n}\n";
  return json.str();
}
}  // namespace

sd::Pointer compileDynamicShapePlan(sd::Pointer serializedPlan, sd::LongType planSize) {
  try {
    if (serializedPlan == nullptr || planSize <= 0) return nullptr;
    return reinterpret_cast<sd::Pointer>(
        NativeDynamicShapePlan::fromSerializedPlan(serializedPlan, planSize));
  } catch (const std::exception& e) {
    setPlanError(-1, e.what());
    return nullptr;
  }
}

int executeDynamicShapePlan(sd::Pointer planHandle, OpaqueContext* opContext,
                            sd::Pointer stream) {
  try {
    if (planHandle == nullptr) {
      setPlanError(1, "executeDynamicShapePlan: null plan handle");
      return 1;
    }
    if (opContext == nullptr) {
      setPlanError(1, "executeDynamicShapePlan: null opContext");
      return 1;
    }

    auto* plan = planOf(planHandle);
    const int numInputs = static_cast<int>(opContext->width());
    const int boundOutputCount = static_cast<int>(opContext->outputWidth());
    const int numOutputs = plan->resolveExecutionOutputCount(boundOutputCount);
    if (numInputs != plan->getNumExternalInputs()) {
      char message[256];
      std::snprintf(message, sizeof(message),
                    "executeDynamicShapePlan: input count mismatch: got %d, expected %d",
                    numInputs, plan->getNumExternalInputs());
      setPlanError(2, message);
      return 2;
    }
    if (numOutputs < 0) {
      char message[256];
      std::snprintf(message, sizeof(message),
                    "executeDynamicShapePlan: output count mismatch: got %d, expected %d",
                    boundOutputCount, plan->getNumRequestedOutputs());
      setPlanError(3, message);
      return 3;
    }

    std::vector<NDArray*> inputs(numInputs);
    for (int i = 0; i < numInputs; ++i) {
      inputs[i] = opContext->array(i);
      if (inputs[i] == nullptr) {
        char message[128];
        std::snprintf(message, sizeof(message),
                      "executeDynamicShapePlan: null input at index %d", i);
        setPlanError(4, message);
        return 4;
      }
      auto* buffer = inputs[i]->dataBuffer();
      if (buffer != nullptr &&
          (buffer->isClosed() || buffer->isDestroyed() || !buffer->isValid())) {
        char message[256];
        std::snprintf(message, sizeof(message),
                      "executeDynamicShapePlan: stale buffer at input %d "
                      "(closed=%d destroyed=%d valid=%d)",
                      i, buffer->isClosed() ? 1 : 0,
                      buffer->isDestroyed() ? 1 : 0,
                      buffer->isValid() ? 1 : 0);
        setPlanError(5, message);
        return 5;
      }
    }

    std::vector<NDArray*> outputs(numOutputs, nullptr);
    void* requestedStream =
        stream != nullptr ? reinterpret_cast<void*>(stream)
                          : plan->getExecutionStream();
    auto* executionStream =
        VulkanExecutionStream::fromOpaque(requestedStream, false);
    if (executionStream == nullptr || !executionStream->isActive()) {
      setPlanError(6, "executeDynamicShapePlan: invalid Vulkan execution stream");
      return 6;
    }

    VulkanExecutionStreamGuard streamGuard(executionStream);
    auto status = plan->execute(inputs.data(), numInputs, outputs.data(),
                                numOutputs,
                                reinterpret_cast<void*>(executionStream));
    if (status != Status::OK) {
      const char* detail =
          sd::LaunchContext::defaultContext()->errorReference()->errorMessage();
      char message[512];
      if (detail != nullptr && detail[0] != '\0') {
        // Root detail precedes wrapper metadata so ErrorReference truncation cannot
        // hide the Vulkan operation that actually failed.
        std::snprintf(message, sizeof(message),
                      "%s [executeDynamicShapePlan returned %s (%d)]",
                      detail, dsp::dspStatusName(status),
                      static_cast<int>(status));
      } else {
        std::snprintf(message, sizeof(message),
                      "executeDynamicShapePlan returned %s (%d) without native "
                      "failure detail; the Vulkan plan path did not set "
                      "LaunchContext::errorReference",
                      dsp::dspStatusName(status), static_cast<int>(status));
      }
      setPlanError(static_cast<int>(status), message);
      return static_cast<int>(status);
    }

    if (!executionStream->synchronize()) {
      setPlanError(7, "executeDynamicShapePlan: Vulkan stream synchronization failed");
      return 7;
    }

    setPlanError(0, "");
    for (int i = 0; i < numOutputs; ++i) {
      if (outputs[i] != nullptr) opContext->setOutputArray(i, outputs[i], false);
    }
    return 0;
  } catch (const std::exception& e) {
    setPlanError(-1, e.what());
    return -1;
  }
}

void freeDynamicShapePlan(sd::Pointer handle) { delete planOf(handle); }

sd::Pointer createNativePlanCache() {
  try {
    return reinterpret_cast<sd::Pointer>(new NativePlanCache());
  } catch (const std::exception& e) {
    setPlanError(-1, e.what());
    return nullptr;
  }
}

void freeNativePlanCache(sd::Pointer handle) {
  if (handle == nullptr) return;
  auto* cache = reinterpret_cast<NativePlanCache*>(handle);
  // Never destroy a cache while a borrower still owns a raw plan handle.
  cache->clear();
  if (cache->pinnedCount() != 0) return;
  delete cache;
}

void clearNativePlanCacheHandle(sd::Pointer handle) {
  if (handle != nullptr) reinterpret_cast<NativePlanCache*>(handle)->clear();
}

sd::Pointer dispatchNativePlan(sd::Pointer cacheHandle, sd::Pointer planBytes,
                               sd::LongType planBytesLen, sd::Pointer outputNames,
                               sd::LongType numOutputs,
                               sd::Pointer phShapeInfoPtrs,
                               sd::LongType numPlaceholders,
                               int graphExecutionMode, int newBorrower) {
  try {
    if (cacheHandle == nullptr || planBytes == nullptr || planBytesLen <= 0)
      return nullptr;
    auto* cache = reinterpret_cast<NativePlanCache*>(cacheHandle);
    auto** namesArray = reinterpret_cast<const char**>(outputNames);
    std::vector<std::string> names;
    names.reserve(numOutputs);
    for (sd::LongType i = 0; i < numOutputs; ++i)
      if (namesArray != nullptr && namesArray[i] != nullptr)
        names.emplace_back(namesArray[i]);
    std::sort(names.begin(), names.end());

    uint64_t hash = 14695981039346656037ULL;
    for (const auto& name : names) {
      for (char c : name) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 1099511628211ULL;
      }
      hash *= 1099511628211ULL;
    }

    NativePlanCache::Key key;
    key.outputSetHash = hash;

    // Preserve CUDA's cache identity sequencing: two different serialized
    // graphs can have identical output, placeholder-shape, and mode keys.
    uint64_t planHash = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const uint8_t*>(planBytes);
    for (sd::LongType i = 0; i < planBytesLen; ++i) {
      planHash ^= bytes[i];
      planHash *= 1099511628211ULL;
    }
    key.planContentHash = planHash;

    auto** shapes = reinterpret_cast<sd::LongType**>(phShapeInfoPtrs);
    key.phShapeContentHash =
        NativePlanCache::hashShapeInfoContents(shapes, numPlaceholders);
    key.phCount = numPlaceholders;
    key.graphExecutionMode = graphExecutionMode;
    key.threadId = std::hash<std::thread::id>{}(std::this_thread::get_id());

    auto factory = [&]() {
      return NativeDynamicShapePlan::fromSerializedPlan(
          planBytes, planBytesLen,
          static_cast<GraphExecutionMode>(graphExecutionMode));
    };
    // Cache hits acquire a lease only for a distinct borrower. Same-executor
    // redispatches pass newBorrower=0 and must not accumulate pins.
    auto* plan = cache->getOrInsert(key, factory, newBorrower != 0);
    if (plan != nullptr && newBorrower != 0)
      plan->invalidateExternalViewSlotsOnReacquire();
    return reinterpret_cast<sd::Pointer>(plan);
  } catch (const std::exception& e) {
    setPlanError(-1, e.what());
    return nullptr;
  }
}

void unpinNativePlan(sd::Pointer cacheHandle, sd::Pointer planHandle) {
  if (cacheHandle != nullptr && planHandle != nullptr)
    reinterpret_cast<NativePlanCache*>(cacheHandle)->unpinPlan(planOf(planHandle));
}

void setPlanCacheShutdownInProgress(bool value) {
  NativePlanCache::setShutdownInProgress(value);
}

void clearDynamicShapePlanCaches(sd::Pointer handle) {
  if (handle != nullptr) planOf(handle)->clearShapeCaches();
}
void clearAllDynamicShapePlanCachesForce(sd::Pointer handle) {
  if (handle != nullptr) planOf(handle)->clearAllShapeCachesForce();
}
int releaseGpuIntermediates(sd::Pointer handle) {
  return handle == nullptr ? 0 : planOf(handle)->releaseGpuIntermediates();
}

int getPlanNumExternalInputs(sd::Pointer h) { return h ? planOf(h)->getNumExternalInputs() : -1; }
const char* getPlanExternalInputName(sd::Pointer h, int i) {
  if (!h || i < 0 || i >= planOf(h)->getNumExternalInputs()) return nullptr;
  const auto& name = planOf(h)->getExternalInputName(i);
  return name.empty() ? nullptr : name.c_str();
}
int getPlanGraphExecutionMode(sd::Pointer h) { return h ? static_cast<int>(planOf(h)->getGraphExecutionMode()) : -1; }
int getPlanNumRequestedOutputs(sd::Pointer h) { return h ? planOf(h)->getNumRequestedOutputs() : -1; }
int getPlanNumSlots(sd::Pointer h) { return h ? planOf(h)->getNumSlots() : -1; }
int getPlanNumSegments(sd::Pointer h) { return h ? static_cast<int>(planOf(h)->getSegments().size()) : -1; }
int getPlanSegmentCount(sd::Pointer h) { return getPlanNumSegments(h); }
int getPlanPhase(sd::Pointer h) { return h ? planOf(h)->getPlanPhaseCode() : -1; }
const char* getPlanLifecycleSnapshot(sd::Pointer h) {
  thread_local std::string result;
  if (h == nullptr) {
    result = "valid=false";
    return result.c_str();
  }
  auto* plan = planOf(h);
  const auto& lifecycle = plan->planLifecycle();
  const auto& segments = plan->getSegments();
  int buildingSegments = 0;
  int sealedSegments = 0;
  int failedSegments = 0;
  for (const auto& segment : segments) {
    switch (segment.exec.graphNodePhase()) {
      case GraphNodePhase::BUILDING: ++buildingSegments; break;
      case GraphNodePhase::SEALED: ++sealedSegments; break;
      case GraphNodePhase::FAILED: ++failedSegments; break;
    }
  }
  result = "valid=true;planPhase=" + std::to_string(lifecycle.toLegacyCode()) +
           ";graphNodePhase=" + std::to_string(static_cast<int>(lifecycle.phase)) +
           ";buildStage=" + std::to_string(static_cast<int>(lifecycle.buildStage)) +
           ";executionCount=" + std::to_string(plan->getExecuteCount()) +
           ";postFreezeExecutionCount=" + std::to_string(lifecycle.postFreezeExecCount) +
           ";pointersStableCount=" + std::to_string(lifecycle.pointersStableCount) +
           ";compilationDone=" + (lifecycle.compilationDone ? "true" : "false") +
           ";segmentCount=" + std::to_string(static_cast<int>(segments.size())) +
           ";buildingSegments=" + std::to_string(buildingSegments) +
           ";sealedSegments=" + std::to_string(sealedSegments) +
           ";failedSegments=" + std::to_string(failedSegments);
  return result.c_str();
}
// Compilation-seal counters are provided by legacy/impl/NativeOps_dsp_shared.cpp.

void setPlanCudaGraphsEnabled(sd::Pointer h, bool v) { if (h) planOf(h)->setCudaGraphsEnabled(v); }
void setPlanShapesFrozen(sd::Pointer h, bool v) { if (h) planOf(h)->setShapesFrozen(v); }
void setPlanShapeOnlyMode(sd::Pointer h, bool v) { if (h) planOf(h)->setShapeOnlyMode(v); }
void setPlanExecutionTimingEnabled(sd::Pointer h, bool v) { if (h) planOf(h)->setExecutionTimingEnabled(v); }
void setPlanJitMode(sd::Pointer h, int mode) {
  if (!h) return;
  auto value = mode == 1 ? JitMode::JIT_ONLY :
               mode == 2 ? JitMode::GRAPH_PLUS_JIT : JitMode::GRAPH_ONLY;
  planOf(h)->setJitMode(value);
}
void setPlanRuntimeCompilationAllowed(sd::Pointer h, bool allowed) {
  if (h) planOf(h)->setRuntimeCompilationAllowed(allowed);
}
void setPlanRuntimeArtifactDirectory(sd::Pointer h, const char* directory) {
  if (h) {
    planOf(h)->setRuntimeArtifactDirectory(
        directory == nullptr ? std::string() : std::string(directory));
  }
}
void setPlanOutputSlotMaxSizes(sd::Pointer h, sd::LongType n,
                               const int* slots, const sd::LongType* sizes) {
  if (h) planOf(h)->setOutputSlotMaxSizes(slots, sizes, static_cast<int>(n));
}
void configurePlanKvScatter(sd::Pointer h, const int* present,
                            const sd::Pointer* staticBuffers,
                            sd::LongType pairs, int dtype,
                            sd::LongType heads, sd::LongType srcSeq,
                            sd::LongType dstSeq, sd::LongType dim,
                            sd::LongType* position) {
  if (h) planOf(h)->configureKvScatter(present,
      reinterpret_cast<NDArray**>(const_cast<sd::Pointer*>(staticBuffers)),
      static_cast<int>(pairs), static_cast<sd::DataType>(dtype), heads,
      srcSeq, dstSeq, dim, position);
}
void resetPlanKvCachePosition(sd::Pointer h, sd::LongType p) { if (h) planOf(h)->resetKvCachePosition(p); }
sd::LongType getPlanKvCachePosition(sd::Pointer h) { return h ? planOf(h)->getKvCachePosition() : -1; }

unsigned long long getPlanReplaySignatureHash(sd::Pointer h, int s) {
  return validSegment(planOf(h), s) ? planOf(h)->getSegments()[s].exec.replaySignatureHash : 0;
}
int getPlanReplayUnitCount(sd::Pointer h, int s) {
  return validSegment(planOf(h), s) ? planOf(h)->getSegments()[s].exec.replayUnitCount : 0;
}
int getSegmentExecutionCount(sd::Pointer h, int s) {
  return validSegment(planOf(h), s) ? planOf(h)->getSegments()[s].exec.executionCount : -1;
}
int getPlanSegmentExecutionCount(sd::Pointer h, int s) { return getSegmentExecutionCount(h, s); }
int getPlanSegmentExecutionPhase(sd::Pointer h, int s) {
  return validSegment(planOf(h), s) ? planOf(h)->getSegments()[s].exec.getExecutionPhaseCode() : -1;
}
int getPlanPointersStable(sd::Pointer h) { return h ? (planOf(h)->arePointersStable() ? 1 : 0) : -1; }
int getPlanFrozenExecutionCount(sd::Pointer h) {
  if (!h || !planOf(h)->isShapesFrozen()) return -1;
  int count = 0;
  for (const auto& segment : planOf(h)->getSegments())
    count = std::max(count, segment.exec.executionCount);
  return count;
}

int getPlanSlotState(sd::Pointer h, int s) { return h ? planOf(h)->getSlotStateCode(s) : -1; }
const char* getPlanSlotOpName(sd::Pointer h, int s) {
  return h && s >= 0 && s < planOf(h)->getNumSlots()
      ? planOf(h)->getSlots()[s].ident.opName.c_str() : "";
}
int getPlanSlotFlags(sd::Pointer h, int s) {
  if (!h || s < 0 || s >= planOf(h)->getNumSlots()) return -1;
  const auto& slot = planOf(h)->getSlots()[s];
  int flags = 0;
  if (slot.isViewCapableOp()) flags |= 1 << 0;
  if (slot.isDataDependent()) flags |= 1 << 1;
  if (slot.flags.outputShapeDependsOnInputValues) flags |= 1 << 2;
  if (slot.isIdentityOp()) flags |= 1 << 3;
  if (slot.flags.inPlaceFused) flags |= 1 << 4;
  if (slot.fusedChain.isFusedChainHead) flags |= 1 << 5;
  if (slot.fusedChain.isFusedChainTail) flags |= 1 << 6;
  if (slot.needsZeroedOutput()) flags |= 1 << 7;
  if (slot.flags.needsIntLongSync) flags |= 1 << 8;
  if (slot.shapeCache.shapeStatic) flags |= 1 << 9;
  if (slot.slotPhase.isSealed() && slot.slotPhase.isConstant) flags |= 1 << 10;
  return flags;
}
int getPlanSlotIOCounts(sd::Pointer h, int s, int* in, int* out) {
  if (!h || s < 0 || s >= planOf(h)->getNumSlots()) return -1;
  if (in) *in = planOf(h)->getSlots()[s].wiring.numInputs;
  if (out) *out = planOf(h)->getSlots()[s].wiring.numOutputs;
  return 0;
}

void markPlanExternalInputVariable(sd::Pointer h, int i) { if (h) planOf(h)->markExternalInputVariable(i); }
void markPlanExternalInputPlaceholder(sd::Pointer h, int i) { if (h) planOf(h)->markExternalInputPlaceholder(i); }
int getPlanNumCachedVariableExtIndices(sd::Pointer h) { return h ? planOf(h)->getNumCachedVariableExtIndices() : 0; }
int getPlanCachedVariableExtIndex(sd::Pointer h, int i) { return h ? planOf(h)->getCachedVariableExtIndex(i) : -1; }
bool getPlanIsExternalInputVariable(sd::Pointer h, int i) { return h && planOf(h)->isExternalInputVariable(i); }
bool getPlanIsExternalInputPlaceholder(sd::Pointer h, int i) { return h && planOf(h)->isExternalInputPlaceholder(i); }
int getPlanNumVariableExternalInputs(sd::Pointer h) { return h ? planOf(h)->getNumVariableExternalInputs() : 0; }
int getPlanExecuteCount(sd::Pointer h) { return h ? planOf(h)->getExecuteCount() : 0; }

void setPlanMinCaptureSegmentSize(sd::Pointer, int) {
  // Segment boundaries are discovered from the compiled graph.
}
void setPlanMaxCaptureSegmentSize(sd::Pointer, int) {
  // Segment boundaries are discovered from the compiled graph.
}
void setPlanTraceEnabled(sd::Pointer h, bool enabled) {
  if (h) planOf(h)->setTraceEnabled(enabled);
}

int getPlanNumCapturedGraphSegments(sd::Pointer h) {
  return h ? planOf(h)->getNumCapturedGraphSegments() : -1;
}
int getPlanTotalGraphReplays(sd::Pointer h) {
  return h ? planOf(h)->getTotalGraphReplays() : -1;
}

bool validatePlanCapturedGraph(sd::Pointer h) {
  if (!h) return false;
  const auto& segments = planOf(h)->getSegments();
  if (segments.empty()) return false;
  for (const auto& segment : segments) {
    if (!segment.def.isCapturable || !segmentHasReadyReplay(segment)) return false;
  }
  return true;
}

int getPlanNumHostOnlyOps(sd::Pointer h) {
  if (!h) return 0;
  int count = 0;
  for (const auto& segment : planOf(h)->getSegments()) {
    if (!segmentHasReadyReplay(segment))
      count += std::max(0, segment.def.endSlot - segment.def.startSlot + 1);
  }
  return count;
}

const char* getPlanHostOnlyOpNames(sd::Pointer h) {
  thread_local std::string result;
  result.clear();
  if (!h) return result.c_str();
  const auto* slots = planOf(h)->getSlots();
  if (!slots) return result.c_str();
  for (const auto& segment : planOf(h)->getSegments()) {
    if (segmentHasReadyReplay(segment)) continue;
    for (int slot = segment.def.startSlot; slot <= segment.def.endSlot; ++slot) {
      if (!result.empty()) result += '|';
      result += slots[slot].ident.opName;
    }
  }
  return result.c_str();
}

void printPlanCapturedGraphDebug(sd::Pointer h) {
  if (!h) return;
  auto* plan = planOf(h);
  std::fprintf(stderr, "Vulkan DSP captured segment debug:\n");
  int index = 0;
  for (const auto& segment : plan->getSegments()) {
    const GraphReplayHandle* replay = segmentRepresentativeReplay(segment);
    const ReplayStatistics stats = segmentReplayStatistics(segment);
    const char* backend = !segment.exec.compiledByBackend.empty()
        ? segment.exec.compiledByBackend.c_str()
        : (replay != nullptr ? replay->backendName() : "");
    std::fprintf(stderr,
        "  segment=%d slots=[%d,%d] capturable=%d state=%d backend=%s "
        "operations=%d replays=%d captureMs=%.3f lastReplayMs=%.3f\n",
        index++, segment.def.startSlot, segment.def.endSlot,
        segment.def.isCapturable ? 1 : 0, segmentReplayState(segment),
        backend, stats.numOperations, stats.replayCount, stats.captureTimeMs,
        stats.lastReplayTimeMs);
  }
  if (plan->getTrace()) plan->getTrace()->dump(stderr, DspExecutionTrace::CAPACITY);
}

const char* getPlanCaptureStats(sd::Pointer h) {
  thread_local std::string result;
  if (!h) {
    result = "null";
    return result.c_str();
  }
  int captured = 0;
  int failed = 0;
  int nonCapturable = 0;
  int capturedSlots = 0;
  int failedSlots = 0;
  int nonCapturableSlots = 0;
  int replays = 0;
  double captureMs = 0.0;
  for (const auto& segment : planOf(h)->getSegments()) {
    const int slots = std::max(0, segment.def.endSlot - segment.def.startSlot + 1);
    if (!segment.def.isCapturable) {
      ++nonCapturable;
      nonCapturableSlots += slots;
    } else if (segmentHasReadyReplay(segment)) {
      ++captured;
      capturedSlots += slots;
      const ReplayStatistics stats = segmentReplayStatistics(segment);
      replays += stats.replayCount;
      captureMs += stats.captureTimeMs;
    } else if (segment.exec.compilationFailed ||
               segmentHasErroredReplay(segment)) {
      ++failed;
      failedSlots += slots;
    }
  }
  std::ostringstream stats;
  stats << "capturedSegments=" << captured
        << " capturedSlots=" << capturedSlots
        << " failedSegments=" << failed
        << " failedSlots=" << failedSlots
        << " nonCapturableSegments=" << nonCapturable
        << " nonCapturableSlots=" << nonCapturableSlots
        << " totalReplays=" << replays
        << " totalCaptureMs=" << captureMs;
  result = stats.str();
  return result.c_str();
}

const char* getPlanSegmentStatisticsJson(sd::Pointer h, int index) {
  thread_local std::string result;
  if (!validSegment(planOf(h), index)) {
    result.clear();
    return result.c_str();
  }
  const auto& segment = planOf(h)->getSegments()[index];
  const GraphReplayHandle* replay = segmentRepresentativeReplay(segment);
  const ReplayStatistics replayStats = segmentReplayStatistics(segment);
  const char* backend = !segment.exec.compiledByBackend.empty()
      ? segment.exec.compiledByBackend.c_str()
      : (replay != nullptr ? replay->backendName() : "");
  int compositeIslands = 0;
  for (const auto& handle :
       segment.exec.compositeReplaySchedule.compositeReplayHandles) {
    if (handle != nullptr && handle->isReady()) ++compositeIslands;
  }
  std::ostringstream json;
  json << "{\"numOperations\":"
       << (segment.def.endSlot - segment.def.startSlot + 1)
       << ",\"replayCount\":" << replayStats.replayCount
       << ",\"replayState\":" << segmentReplayState(segment)
       << ",\"backendName\":\"" << jsonEscape(backend) << "\""
       << ",\"executionCount\":" << segment.exec.executionCount
       << ",\"capturable\":" << (segment.def.isCapturable ? "true" : "false")
       << ",\"compilationFailed\":"
       << (segment.exec.compilationFailed ? "true" : "false")
       << ",\"mergedGroups\":"
       << segment.exec.compositeReplaySchedule.mergedReplayHandles.size()
       << ",\"compositeIslands\":" << compositeIslands
       << ",\"captureTimeMs\":" << replayStats.captureTimeMs
       << ",\"lastReplayTimeMs\":" << replayStats.lastReplayTimeMs
       << ",\"estimatedMemory\":" << replayStats.estimatedMemory
       << ",\"deviceName\":\"" << jsonEscape(replayStats.deviceName) << "\""
       << ",\"apiVersion\":" << replayStats.apiVersion
       << ",\"memoryBudgetBytes\":" << replayStats.memoryBudgetBytes << '}';
  result = json.str();
  return result.c_str();
}

bool isPlanSegmentCapturable(sd::Pointer h, int index) {
  return validSegment(planOf(h), index) &&
         planOf(h)->getSegments()[index].def.isCapturable;
}
bool isPlanSegmentCaptureFailed(sd::Pointer h, int index) {
  if (!validSegment(planOf(h), index)) return false;
  const auto& segment = planOf(h)->getSegments()[index];
  return segment.exec.compilationFailed || segmentHasErroredReplay(segment);
}

int dspValidateOutputs(sd::Pointer h, int* flagsOut) {
  if (!h || !flagsOut) return -1;
  auto* plan = planOf(h);
  const int count = plan->getNumRequestedOutputs();
  if (count <= 0) return 0;
  const auto* definition = plan->getPlanDefinition();
  auto** outputSlots = plan->getOutputSlots();
  if (!definition || !outputSlots) return -1;
  std::vector<NDArray*> outputs(count);
  for (int i = 0; i < count; ++i) {
    const int slot = definition->requestedOutputSlotIndices()[i];
    outputs[i] = slot >= 0 && slot < plan->getTotalOutputSlots()
        ? outputSlots[slot] : nullptr;
  }
  return sd::graph::dspValidateOutputs(outputs.data(), count, flagsOut);
}

int dspDetectStaleOutputs(sd::Pointer h, float* prevNorms, bool* staleOut,
                          float epsilon) {
  if (!h || !prevNorms || !staleOut) return -1;
  auto* plan = planOf(h);
  const int count = plan->getNumRequestedOutputs();
  if (count <= 0) return 0;
  const auto* definition = plan->getPlanDefinition();
  auto** outputSlots = plan->getOutputSlots();
  if (!definition || !outputSlots) return -1;
  std::vector<NDArray*> outputs(count);
  for (int i = 0; i < count; ++i) {
    const int slot = definition->requestedOutputSlotIndices()[i];
    outputs[i] = slot >= 0 && slot < plan->getTotalOutputSlots()
        ? outputSlots[slot] : nullptr;
  }
  return sd::graph::dspDetectStaleOutputs(
      outputs.data(), count, prevNorms, staleOut, epsilon);
}

const char* getPlanSegmentsSummaryJson(sd::Pointer h) {
  thread_local std::string result;
  if (!h) {
    result = "[]";
    return result.c_str();
  }
  const auto* slots = planOf(h)->getSlots();
  std::ostringstream json;
  json << '[';
  bool firstSegment = true;
  int index = 0;
  for (const auto& segment : planOf(h)->getSegments()) {
    if (!firstSegment) json << ',';
    firstSegment = false;
    std::unordered_map<std::string, int> opCounts;
    if (slots) {
      for (int slot = segment.def.startSlot; slot <= segment.def.endSlot; ++slot)
        ++opCounts[slots[slot].ident.opName];
    }
    json << "{\"index\":" << index++
         << ",\"startSlot\":" << segment.def.startSlot
         << ",\"endSlot\":" << segment.def.endSlot
         << ",\"numOps\":"
         << (segment.def.endSlot - segment.def.startSlot + 1)
         << ",\"executionCount\":" << segment.exec.executionCount
         << ",\"isCapturable\":"
         << (segment.def.isCapturable ? "true" : "false")
         << ",\"compilationFailed\":"
         << (segment.exec.compilationFailed ? "true" : "false")
         << ",\"hasReplayHandle\":"
         << (segmentHasReadyReplay(segment) ? "true" : "false")
         << ",\"backendName\":\""
         << jsonEscape(segment.exec.replayHandle
                           ? segment.exec.replayHandle->backendName()
                           : segment.exec.compiledByBackend)
         << "\",\"ops\":{";
    bool firstOp = true;
    for (const auto& op : opCounts) {
      if (!firstOp) json << ',';
      firstOp = false;
      json << '\"' << jsonEscape(op.first) << "\":" << op.second;
    }
    json << "}}";
  }
  json << ']';
  result = json.str();
  return result.c_str();
}

const char* getPlanCudaGraphChromeTraceJson(sd::Pointer h) {
  thread_local std::string result;
  result = planTraceJson(planOf(h));
  return result.c_str();
}

void clearPlanCudaGraphTimeline(sd::Pointer h) {
  if (h && planOf(h)->getTrace()) planOf(h)->getTrace()->reset();
}

bool exportPlanCudaGraphChromeTrace(sd::Pointer h, const char* outputPath) {
  if (!h || !outputPath) return false;
  std::ofstream output(outputPath);
  if (!output.is_open()) return false;
  output << planTraceJson(planOf(h));
  return output.good();
}

bool exportPlanCudaGraphHtml(sd::Pointer h, const char* outputPath) {
  if (!h || !outputPath) return false;
  std::ofstream output(outputPath);
  if (!output.is_open()) return false;
  output << "<!doctype html><meta charset=\"utf-8\">"
            "<title>Vulkan DSP execution trace</title>"
            "<pre id=\"trace\"></pre><script>"
            "const trace="
         << planTraceJson(planOf(h))
         << ";document.getElementById('trace').textContent="
            "JSON.stringify(trace,null,2);</script>";
  return output.good();
}

bool debugDumpPlanCudaGraph(sd::Pointer h, const char* outputPath) {
  if (!h || !outputPath) return false;
  const std::string base(outputPath);
  const bool jsonOk =
      exportPlanCudaGraphChromeTrace(h, (base + ".json").c_str());
  FILE* text = std::fopen((base + ".txt").c_str(), "w");
  if (!text) return false;
  auto* trace = planOf(h)->getTrace();
  if (trace) trace->dump(text, DspExecutionTrace::CAPACITY);
  const bool textOk = std::fclose(text) == 0;
  return jsonOk && textOk;
}

const char* getPlanSegmentBackendName(sd::Pointer h, int s) {
  if (!validSegment(planOf(h), s)) return "";
  const auto& segment = planOf(h)->getSegments()[s];
  if (!segment.exec.compiledByBackend.empty()) {
    return segment.exec.compiledByBackend.c_str();
  }
  const GraphReplayHandle* replay = segmentRepresentativeReplay(segment);
  return replay != nullptr ? replay->backendName() : "";
}
const char* getPlanAvailableBackends(sd::Pointer) {
  thread_local std::string result;
  auto& manager = VulkanDeviceManager::getInstance();
  const bool runtimeAvailable = manager.initialize() && manager.deviceCount() > 0;
#if defined(HAVE_MLIR) && HAVE_MLIR
  const bool compiledCapability = true;
#else
  const bool compiledCapability = false;
#endif
  std::ostringstream json;
  json << "[{\"name\":\"Vulkan\",\"type\":\"GPU\",\"available\":"
       << (runtimeAvailable && compiledCapability ? "true" : "false")
       << ",\"priority\":0,\"deviceCount\":" << manager.deviceCount()
       << ",\"mlirCompiled\":"
       << (compiledCapability ? "true" : "false") << "}]";
  result = json.str();
  return result.c_str();
}
const char* getPlanSegmentCompiledBackend(sd::Pointer h, int s) {
  return validSegment(planOf(h), s)
      ? planOf(h)->getSegments()[s].exec.compiledByBackend.c_str() : "";
}
const char* getPlanSegmentCompilationAudit(sd::Pointer h, int s) {
  thread_local std::string result;
  result = validSegment(planOf(h), s) ? planOf(h)->getSegmentCompilationAudit(s) : "{}";
  return result.c_str();
}
void invalidatePlanSegmentCache(sd::Pointer h, int s) {
  if (!validSegment(planOf(h), s)) return;
  auto& exec = planOf(h)->getSegmentsMutable()[s].exec;
  exec.replayHandle.reset();
  SegmentLifecycle::resetForCacheInvalidation(exec);
  exec.cachedShapeKey = 0;
  exec.executionCount = 0;
  exec.compiledByBackend.clear();
}
void invalidatePlanBackendCaches(sd::Pointer h, const char* name) {
  if (!h) return;
  const std::string requested = name ? name : "";
  for (auto& segment : planOf(h)->getSegmentsMutable()) {
    if (requested.empty() || segment.exec.compiledByBackend == requested) {
      segment.exec.replayHandle.reset();
      SegmentLifecycle::resetForCacheInvalidation(segment.exec);
      segment.exec.cachedShapeKey = 0;
      segment.exec.executionCount = 0;
      segment.exec.compiledByBackend.clear();
    }
  }
}
const char* getPlanBackendCacheStats(sd::Pointer h) {
  thread_local std::string result;
  if (!h) { result = "{}"; return result.c_str(); }
  std::map<std::string, int> counts;
  for (const auto& segment : planOf(h)->getSegments())
    if (!segment.exec.compiledByBackend.empty()) ++counts[segment.exec.compiledByBackend];
  std::ostringstream json;
  json << "{\"backends\":[";
  bool first = true;
  for (const auto& entry : counts) {
    if (!first) json << ',';
    first = false;
    json << "{\"name\":\"" << entry.first << "\",\"compiledSegments\":" << entry.second << '}';
  }
  json << "]}";
  result = json.str();
  return result.c_str();
}
void setPlanSegmentBackendOverride(sd::Pointer h, int s, const char* name) {
  if (validSegment(planOf(h), s))
    planOf(h)->getSegmentsMutable()[s].def.backendOverride = name ? name : "";
}
void setPlanBackendPriority(sd::Pointer h, const char* value) {
  if (!h || !value) return;
  std::vector<std::string> priority;
  std::stringstream input(value);
  std::string backend;
  while (std::getline(input, backend, ',')) {
    if (!backend.empty()) priority.push_back(backend);
  }
  planOf(h)->setBackendPriority(priority);
}

int executeFrozenPlan(sd::Pointer h, OpaqueContext* c, sd::Pointer s) {
  return executeDynamicShapePlan(h, c, s);
}
int isFrozenPlanSealed(sd::Pointer h) {
  return h ? (planOf(h)->getPlanPhaseCode() >= 3 ? 1 : 0) : -1;
}
int getFrozenPlanBuildPassCount(sd::Pointer h) {
  if (!h) return -1;
  const int phase = planOf(h)->getPlanPhaseCode();
  return phase >= 3 ? 2 : phase;
}
int getSegmentExecutorPhase(sd::Pointer h, int s) {
  const int phase = getPlanSegmentExecutionPhase(h, s);
  switch (phase) {
    case 0:
    case 1:
    case 2:
      return 0;
    case 3:
      return 1;
    case 4:
      return 2;
    default:
      return -1;
  }
}

void dspDiagSetCategories(int m) { DspDiagnostics::getInstance().setCategories(static_cast<uint32_t>(m)); }
void dspDiagEnableCategories(int m) { DspDiagnostics::getInstance().enableCategories(static_cast<uint32_t>(m)); }
void dspDiagDisableCategories(int m) { DspDiagnostics::getInstance().disableCategories(static_cast<uint32_t>(m)); }
int dspDiagGetEnabledMask() { return static_cast<int>(DspDiagnostics::getInstance().getEnabledMask()); }
void dspDiagSetLevel(int l) { DspDiagnostics::getInstance().setLevel(static_cast<DspDiagLevel>(l)); }
int dspDiagGetLevel() { return static_cast<int>(DspDiagnostics::getInstance().getLevel()); }
void dspDiagSetJsonPath(const char* p) { if (p) DspDiagnostics::getInstance().setJsonPath(p); }
void dspDiagRecordJavaEvent(int c, int slot, int seg, const char* op,
                            sd::LongType us, const char* msg) {
  DspDiagnostics::getInstance().recordEvent(static_cast<uint32_t>(c), slot, seg,
      -1, op, static_cast<int64_t>(us), "%s", msg ? msg : "");
}
const char* dspDiagGetPlanReport() {
  thread_local std::string value;
  value = DspDiagnostics::getInstance().generatePlanReport();
  return value.c_str();
}
const char* dspDiagGetJsonReport() {
  thread_local std::string value;
  value = DspDiagnostics::getInstance().generateJsonReport();
  return value.c_str();
}
void dspDiagClear() { DspDiagnostics::getInstance().clear(); }
int dspDiagGetStepCount() { return DspDiagnostics::getInstance().getStepsExecuted(); }
long long dspDiagGetTotalEventCount() { return DspDiagnostics::getInstance().getTotalEventCount(); }
long long dspDiagGetCategoryEventCount(int i) { return DspDiagnostics::getInstance().getCategoryEventCount(i); }

void setDspFreezeMergeSegments(bool v) { sd::Environment::getInstance().setDspFreezeMergeSegments(v); }
void setDspFreezeRecompile(bool v) { sd::Environment::getInstance().setDspFreezeRecompile(v); }
bool getDspFreezeMergeSegments() { return sd::Environment::getInstance().dspFreezeMergeSegments(); }
bool getDspFreezeRecompile() { return sd::Environment::getInstance().dspFreezeRecompile(); }

int getPlanNumStagingBuffers(sd::Pointer h) { return h ? planOf(h)->getNumStagingBuffers() : 0; }
long long getPlanStagingBufferAddress(sd::Pointer h, int i) { return h ? planOf(h)->getStagingBufferAddress(i) : 0; }
long long getPlanEffectiveExternalAddress(sd::Pointer h, int i) { return h ? planOf(h)->getEffectiveExternalAddress(i) : 0; }
long long getPlanLastExternalInputAddress(sd::Pointer h, int i) { return h ? planOf(h)->getLastExternalInputAddress(i) : 0; }
OpaqueNDArray getPlanStagingBufferArray(sd::Pointer h, int i) { return h ? planOf(h)->getStagingBufferArray(i) : nullptr; }
int copyPlanStagingToBuffer(sd::Pointer h, int i, OpaqueDataBuffer* dst) {
  if (h == nullptr) return -1;
  if (dst == nullptr) return -3;
  return planOf(h)->copyStagingToBuffer(i, dst->dataBuffer());
}
OpaqueNDArray getPlanSlotOutputArray(sd::Pointer h, int i) {
  if (!h || i < 0 || i >= planOf(h)->getNumSlots()) return nullptr;
  return planOf(h)->getSlotOutputArray(i);
}
int getTotalPlanOutputSlots(sd::Pointer h) { return h ? planOf(h)->getTotalOutputSlots() : 0; }
int getPlanSlotGeneration(sd::Pointer h, int i) { return h ? planOf(h)->getSlotGeneration(i) : -1; }
int getPlanSegmentReplayMode(sd::Pointer h, int i) { return h ? planOf(h)->getSegmentReplayMode(i) : 0; }
long long getPlanSegmentArgGeneration(sd::Pointer h, int i) { return h ? planOf(h)->getSegmentArgGeneration(i) : -1; }
long long getPlanSegmentCapturedArgGeneration(sd::Pointer h, int i) { return h ? planOf(h)->getSegmentCapturedArgGeneration(i) : -1; }
int getPlanSegmentNeedsArgRefresh(sd::Pointer h, int i) { return h ? planOf(h)->getSegmentNeedsArgRefresh(i) : 0; }
long long getPlanSegmentCapturedInputAddrKey(sd::Pointer h, int i) { return h ? planOf(h)->getSegmentCapturedInputAddrKey(i) : 0; }

int getLastExecSegmentsWarmup(sd::Pointer h) { return h ? planOf(h)->getLastExecSegmentsWarmup() : -1; }
int getLastExecSegmentsCaptured(sd::Pointer h) { return h ? planOf(h)->getLastExecSegmentsCaptured() : -1; }
int getLastExecSegmentsReplayed(sd::Pointer h) { return h ? planOf(h)->getLastExecSegmentsReplayed() : -1; }
int getLastExecSegmentsSlotBySlot(sd::Pointer h) { return h ? planOf(h)->getLastExecSegmentsSlotBySlot() : -1; }
int getLastExecSegmentsFailed(sd::Pointer h) { return h ? planOf(h)->getLastExecSegmentsFailed() : -1; }
int getLastExecSegmentsTotal(sd::Pointer h) { return h ? planOf(h)->getLastExecSegmentsTotal() : -1; }
int getLastExecSyncLevel(sd::Pointer h) { return h ? planOf(h)->getLastExecSyncLevel() : -1; }
int getLastExecStreamSyncCount(sd::Pointer h) { return h ? planOf(h)->getLastExecStreamSyncCount() : -1; }
int getLastExecConsecutiveUnchangedCount(sd::Pointer h) { return h ? planOf(h)->getLastExecConsecutiveUnchangedCount() : -1; }

bool getPlanBufferColoringApplied(sd::Pointer h) { return h && planOf(h)->bufferColorMap().isApplied(); }
int getPlanBufferColoringNumColors(sd::Pointer h) { return h ? planOf(h)->bufferColorMap().numColors() : 0; }
sd::LongType getPlanBufferColoringBytesSaved(sd::Pointer h) {
  return h ? static_cast<sd::LongType>(planOf(h)->bufferColorMap().estimatedBytesSaved()) : 0;
}
int getPlanSlotColor(sd::Pointer h, int i) { return h ? planOf(h)->bufferColorMap().colorOf(i) : -1; }


#endif  // HAVE_VULKAN
